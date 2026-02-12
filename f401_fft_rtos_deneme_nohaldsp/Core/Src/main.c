
// STM32F401 Computing Node
// First Release: 01/02/2026 version: 0.1

#include "main.h"
#include "cmsis_os.h"
#include "arm_math.h"
#include "wizchip_conf.h"
#include "socket.h"
#include <string.h>
#include <stdio.h>

SPI_HandleTypeDef hspi1;
DMA_HandleTypeDef hdma_spi1_rx;
DMA_HandleTypeDef hdma_spi1_tx;

osThreadId TaskRXHandle;
osThreadId TaskDSPHandle;
osThreadId TaskTX_F411Handle;
osThreadId TaskTX_PCHandle;

osMessageQId rawDataQueue;
osMessageQId f411Queue;
osMessageQId pcFFTQueue;

osSemaphoreId spiMutex;
osSemaphoreId dmaTxSem;
osSemaphoreId dmaRxSem;

// ==================== NETWORK CONFIG ====================
#define SOCK_SIMULINK   0
#define SOCK_F411       1
#define SOCK_PC_FFT     2

#define PORT_SIMULINK   5008
#define PORT_PC_FFT     5007
#define PORT_F411_DEST  6000

uint8_t destIP_PC[4]   = {192, 168, 1, 103};
uint8_t destIP_F411[4] = {192, 168, 1, 107};

wiz_NetInfo netInfo = {
    .mac  = {0x00, 0x08, 0xDC, 0xAB, 0xCD, 0xEF},
    .ip   = {192, 168, 1, 100},
    .sn   = {255, 255, 255, 0},
    .gw   = {192, 168, 1, 1},
    .dns  = {8, 8, 8, 8},
    .dhcp = NETINFO_STATIC
};

// ==================== DATA STRUCTURES ====================
#define BATCH_SIZE 50

typedef struct {
    float32_t data[6];
} RawSample_t;

typedef struct {
    uint32_t header;
    RawSample_t samples[BATCH_SIZE];
} BatchPacket_t;

#define RX_POOL_SIZE 2
typedef struct {
    BatchPacket_t packet;
    volatile uint8_t is_locked;
} DataBlock_t;

DataBlock_t rxPool[RX_POOL_SIZE];

// --- F411 Data Types ---
typedef struct {
    uint32_t header; // 0x1111AAAA
    float trq[BATCH_SIZE];
    float rpm[BATCH_SIZE];
} FastData_t;

#define F411_FFT_SIZE 160
typedef struct {
    uint32_t header; // 0x2222BBBB
    float fft_partial[F411_FFT_SIZE];
} SlowData_t;

typedef struct {
    uint8_t type; // 0: Fast, 1: Slow
    union {
        FastData_t fast;
        SlowData_t slow;
    } payload;
} F411_Queue_Item_t;

// ==================== DSP BUFFERS ====================
#define FFT_SIZE 2048
#define FFT_OUTPUT_SIZE 1024

float32_t iq_buf[FFT_SIZE];
float32_t fft_mag_iq[FFT_OUTPUT_SIZE];
float32_t hanning_window[FFT_SIZE];

arm_rfft_fast_instance_f32 fft_instance;

// ==================== FUNCTIONS ====================
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_SPI1_Init(void);
static void enable_fpu(void);

void StartTaskRX(void const * argument);
void StartTaskDSP(void const * argument);
void StartTaskTX_F411(void const * argument);
void StartTaskTX_PC(void const * argument);

// W5500 Callbacks
void W5500_Select(void) { HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET); }
void W5500_Deselect(void) { HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET); }
void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi) { osSemaphoreRelease(dmaTxSem); }
void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi) { osSemaphoreRelease(dmaRxSem); }
void W5500_ReadBuff_OS(uint8_t* buff, uint16_t len) {
    HAL_SPI_Receive_DMA(&hspi1, buff, len);
    osSemaphoreWait(dmaRxSem, osWaitForever);
}
void W5500_WriteBuff_OS(uint8_t* buff, uint16_t len) {
    HAL_SPI_Transmit_DMA(&hspi1, buff, len);
    osSemaphoreWait(dmaTxSem, osWaitForever);
}
uint8_t W5500_ReadByte(void) { uint8_t b; HAL_SPI_Receive(&hspi1, &b, 1, 10); return b; }
void W5500_WriteByte(uint8_t b) { HAL_SPI_Transmit(&hspi1, &b, 1, 10); }

// Transform Functions
void clarke_transform(float32_t ia, float32_t ib, float32_t ic, float32_t *alpha, float32_t *beta) {
    *alpha = ia;
    *beta = (ia + 2.0f * ib) / 1.7320508f;
}
void park_transform(float32_t alpha, float32_t beta, float32_t angle, float32_t *id, float32_t *iq) {
    float32_t cos_theta = arm_cos_f32(angle);
    float32_t sin_theta = arm_sin_f32(angle);
    *id = alpha * cos_theta + beta * sin_theta;
    *iq = -alpha * sin_theta + beta * cos_theta;
}

int main(void) {

    enable_fpu();
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_SPI1_Init();

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, GPIO_PIN_RESET);
    HAL_Delay(10);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, GPIO_PIN_SET);
    HAL_Delay(100);

    reg_wizchip_cs_cbfunc(W5500_Select, W5500_Deselect);
    reg_wizchip_spi_cbfunc(W5500_ReadByte, W5500_WriteByte);
    reg_wizchip_spiburst_cbfunc(W5500_ReadBuff_OS, W5500_WriteBuff_OS);

    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);

    for(int i=0; i<FFT_SIZE; i++) {
        hanning_window[i] = 0.5f * (1.0f - arm_cos_f32(2.0f * PI * (float32_t)i / (float32_t)(FFT_SIZE - 1)));
    }

    osSemaphoreDef(SEM_SPI); spiMutex = osSemaphoreCreate(osSemaphore(SEM_SPI), 1);
    osSemaphoreDef(SEM_DMATX); dmaTxSem = osSemaphoreCreate(osSemaphore(SEM_DMATX), 1); osSemaphoreWait(dmaTxSem, 0);
    osSemaphoreDef(SEM_DMARX); dmaRxSem = osSemaphoreCreate(osSemaphore(SEM_DMARX), 1); osSemaphoreWait(dmaRxSem, 0);

    osMessageQDef(Q_RAW, 2, uint32_t); rawDataQueue = osMessageCreate(osMessageQ(Q_RAW), NULL);

    
    // FreeRTOS queue create (Item Size = sizeof(F411_Queue_Item_t))
    osMessageQDef(Q_F411, 4, F411_Queue_Item_t);
    f411Queue = osMessageCreate(osMessageQ(Q_F411), NULL);

    osMessageQDef(Q_PC, 2, uint32_t); pcFFTQueue = osMessageCreate(osMessageQ(Q_PC), NULL);

    for(int i=0; i<RX_POOL_SIZE; i++) rxPool[i].is_locked = 0;

    osThreadDef(NetRX, StartTaskRX, osPriorityHigh, 0, 512);
    TaskRXHandle = osThreadCreate(osThread(NetRX), NULL);

    // DSP task stack size upgraded (FFT + Queue Copy icin)
    osThreadDef(DSP, StartTaskDSP, osPriorityAboveNormal, 0, 1024);
    TaskDSPHandle = osThreadCreate(osThread(DSP), NULL);

    osThreadDef(TxF411, StartTaskTX_F411, osPriorityNormal, 0, 1024);
    TaskTX_F411Handle = osThreadCreate(osThread(TxF411), NULL);

    osThreadDef(TxPC, StartTaskTX_PC, osPriorityLow, 0, 512);
    TaskTX_PCHandle = osThreadCreate(osThread(TxPC), NULL);

    osKernelStart();
    while (1) {}
}

//---------------------- TASK 1 -----------------------------
void StartTaskRX(void const * argument) {
    osSemaphoreWait(spiMutex, osWaitForever);
    // RX: 8KB, PC TX: 4KB
    uint8_t memsize[2][8] = {{8,2,2,2,2,2,2,0}, {2,2,4,2,2,2,2,0}};
    ctlwizchip(CW_INIT_WIZCHIP, (void*)memsize);
    ctlnetwork(CN_SET_NETINFO, (void*)&netInfo);

    socket(SOCK_SIMULINK, Sn_MR_TCP, PORT_SIMULINK, 0);
    listen(SOCK_SIMULINK);
    osSemaphoreRelease(spiMutex);

    uint32_t batch_size_bytes = sizeof(BatchPacket_t);
    uint8_t has_data = 0;

    for(;;) {
        osSemaphoreWait(spiMutex, osWaitForever);
        uint8_t status = getSn_SR(SOCK_SIMULINK);

        if(status == SOCK_ESTABLISHED) {
             has_data = 0;
             while(getSn_RX_RSR(SOCK_SIMULINK) >= batch_size_bytes) {
                 has_data = 1;
                 int idx = -1;
                 for(int i=0; i<RX_POOL_SIZE; i++) if(!rxPool[i].is_locked) { idx = i; break; }

                 if(idx != -1) {
                     rxPool[idx].is_locked = 1;
                     recv(SOCK_SIMULINK, (uint8_t*)&rxPool[idx].packet, batch_size_bytes);
                     if(rxPool[idx].packet.header == 0x42AA0000) {
                        osMessagePut(rawDataQueue, (uint32_t)&rxPool[idx], 0);
                     } else {
                        rxPool[idx].is_locked = 0;
                     }
                 } else {
                     break;
                 }
                 osSemaphoreRelease(spiMutex);
                 osThreadYield();
                 osSemaphoreWait(spiMutex, osWaitForever);
             }
        }
        else if(status == SOCK_CLOSE_WAIT) disconnect(SOCK_SIMULINK);
        else if(status == SOCK_CLOSED) {
            socket(SOCK_SIMULINK, Sn_MR_TCP, PORT_SIMULINK, 0);
            listen(SOCK_SIMULINK);
        }

        osSemaphoreRelease(spiMutex);
        if(!has_data) osDelay(1); else osThreadYield();
    }
}

//------------------------- TASK 2 ------------------------------
void StartTaskDSP(void const * argument) {
    DataBlock_t *currentBlock;
    float32_t id_val, iq_val, alpha, beta;
    uint16_t buf_idx = 0;

    // local variable in stack
    F411_Queue_Item_t f411_item;

    for(;;) {
        osEvent evt = osMessageGet(rawDataQueue, osWaitForever);
        if(evt.status == osEventMessage) {
            currentBlock = (DataBlock_t*)evt.value.p;

            // --- 1. fast data preparing  ---
            f411_item.type = 0; // Fast
            f411_item.payload.fast.header = 0x1111AAAA;
            for(int i=0; i<BATCH_SIZE; i++) {
                f411_item.payload.fast.trq[i] = currentBlock->packet.samples[i].data[4];
                f411_item.payload.fast.rpm[i] = currentBlock->packet.samples[i].data[5];
            }

            xQueueSend(f411Queue, &f411_item, 0);
            

            // --- 2. Clarke-Park Transform ---
            for(int i=0; i<BATCH_SIZE; i++) {
                RawSample_t *r_samples = &currentBlock->packet.samples[i];
                clarke_transform(r_samples->data[0], r_samples->data[1], r_samples->data[2], &alpha, &beta);
                park_transform(alpha, beta, r_samples->data[3], &id_val, &iq_val);

                iq_buf[buf_idx++] = iq_val;

                if(buf_idx >= FFT_SIZE) {
                    buf_idx = 0;
                    arm_mult_f32(iq_buf, hanning_window, iq_buf, FFT_SIZE);
                    arm_rfft_fast_f32(&fft_instance, iq_buf, iq_buf, 0);
                    arm_cmplx_mag_f32(iq_buf, fft_mag_iq, FFT_OUTPUT_SIZE);

                    osMessagePut(pcFFTQueue, (uint32_t)fft_mag_iq, 0);

                    // --- Slow Data (FFT) ---
                    F411_Queue_Item_t f411_fft;
                    f411_fft.type = 1; // Slow
                    f411_fft.payload.slow.header = 0x2222BBBB;
                    memcpy(f411_fft.payload.slow.fft_partial, fft_mag_iq, F411_FFT_SIZE*4);

                    xQueueSend(f411Queue, &f411_fft, 0);
                }
            }

            taskENTER_CRITICAL();
            currentBlock->is_locked = 0;
            taskEXIT_CRITICAL();
        }
    }
}

//---------------------------- TASK 3 ------------------------
void StartTaskTX_F411(void const * argument) {
    osSemaphoreWait(spiMutex, osWaitForever);
    socket(SOCK_F411, Sn_MR_TCP, 3000, 0);
    osSemaphoreRelease(spiMutex);

    F411_Queue_Item_t txItem; 
    uint32_t last_retry = 0;

    for(;;) {
        
        if(xQueueReceive(f411Queue, &txItem, osWaitForever) == pdTRUE) {

            uint8_t* pData;
            int total_len;

            if(txItem.type == 0) {
                pData = (uint8_t*)&txItem.payload.fast;
                total_len = sizeof(FastData_t);
            } else {
                pData = (uint8_t*)&txItem.payload.slow;
                total_len = sizeof(SlowData_t);
            }

            int sent = 0;
            while(sent < total_len) {
                osSemaphoreWait(spiMutex, osWaitForever);
                uint8_t s = getSn_SR(SOCK_F411);

                if(s == SOCK_ESTABLISHED) {
                    uint16_t c = 1024;
                    if(total_len - sent < c) c = total_len - sent;
                    if(getSn_TX_FSR(SOCK_F411) >= c) {
                        send(SOCK_F411, pData + sent, c);
                        sent += c;
                    }
                }
                else if(s == SOCK_CLOSED || s == SOCK_INIT) {
                    if(HAL_GetTick() - last_retry > 1000) {
                        socket(SOCK_F411, Sn_MR_TCP, 3000, 0);
                        connect(SOCK_F411, destIP_F411, PORT_F411_DEST);
                        last_retry = HAL_GetTick();
                    }
                    if(sent > 0) break; else osDelay(5);
                }
                else if(s == SOCK_CLOSE_WAIT) {
                    disconnect(SOCK_F411); close(SOCK_F411);
                }
                osSemaphoreRelease(spiMutex);
                osDelay(1);
            }
        }
    }
}

//-------------------------- TASK 4 -----------------------------
void StartTaskTX_PC(void const * argument) {
    float32_t* pFFTData;
    uint32_t last_retry = 0;
    osSemaphoreWait(spiMutex, osWaitForever);
    socket(SOCK_PC_FFT, Sn_MR_TCP, 3001, 0);
    osSemaphoreRelease(spiMutex);

    for(;;) {
        osEvent evt = osMessageGet(pcFFTQueue, osWaitForever);
        if(evt.status == osEventMessage) {
            pFFTData = (float32_t*)evt.value.p;
            int sent = 0;
            int total = FFT_OUTPUT_SIZE * 4;
            uint8_t* pRaw = (uint8_t*)pFFTData;

            while(sent < total) {
                osSemaphoreWait(spiMutex, osWaitForever);
                uint8_t s = getSn_SR(SOCK_PC_FFT);

                if(s == SOCK_ESTABLISHED) {
                    uint16_t c = 1024;
                    if(total - sent < c) c = total - sent;
                    if(getSn_TX_FSR(SOCK_PC_FFT) >= c) {
                        send(SOCK_PC_FFT, pRaw + sent, c);
                        sent += c;
                    }
                }
                else if(s == SOCK_CLOSED || s == SOCK_INIT) {
                    if(HAL_GetTick() - last_retry > 1000) {
                        socket(SOCK_PC_FFT, Sn_MR_TCP, 3001, 0);
                        connect(SOCK_PC_FFT, destIP_PC, PORT_PC_FFT);
                        last_retry = HAL_GetTick();
                    }
                    if(sent > 0) break;
                }
                else if(s == SOCK_CLOSE_WAIT) {
                    disconnect(SOCK_PC_FFT); close(SOCK_PC_FFT);
                }
                osSemaphoreRelease(spiMutex);
                osDelay(1);
            }
        }
    }
}

//-------------------- HAL INIT ------------------------

static void enable_fpu(void) { SCB->CPACR |= (0xF << 20); }
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 84;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) Error_Handler();
}

static void MX_SPI1_Init(void) {
    hspi1.Instance = SPI1;
    hspi1.Init.Mode = SPI_MODE_MASTER;
    hspi1.Init.Direction = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
    hspi1.Init.NSS = SPI_NSS_SOFT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    if (HAL_SPI_Init(&hspi1) != HAL_OK) Error_Handler();

    hdma_spi1_rx.Instance = DMA2_Stream0;
    hdma_spi1_rx.Init.Channel = DMA_CHANNEL_3;
    hdma_spi1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_spi1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_spi1_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_spi1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_spi1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_spi1_rx.Init.Mode = DMA_NORMAL;
    hdma_spi1_rx.Init.Priority = DMA_PRIORITY_HIGH;
    if (HAL_DMA_Init(&hdma_spi1_rx) != HAL_OK) Error_Handler();
    __HAL_LINKDMA(&hspi1, hdmarx, hdma_spi1_rx);

    hdma_spi1_tx.Instance = DMA2_Stream3;
    hdma_spi1_tx.Init.Channel = DMA_CHANNEL_3;
    hdma_spi1_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_spi1_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_spi1_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_spi1_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_spi1_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_spi1_tx.Init.Mode = DMA_NORMAL;
    hdma_spi1_tx.Init.Priority = DMA_PRIORITY_LOW;
    if (HAL_DMA_Init(&hdma_spi1_tx) != HAL_OK) Error_Handler();
    __HAL_LINKDMA(&hspi1, hdmatx, hdma_spi1_tx);
}

static void MX_DMA_Init(void) {
  __HAL_RCC_DMA2_CLK_ENABLE();
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
  HAL_NVIC_SetPriority(DMA2_Stream3_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream3_IRQn);
}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0|GPIO_PIN_4, GPIO_PIN_SET);
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_4; // RESET & CS
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

void StartDefaultTask(void const * argument) { for(;;) osDelay(1); }
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) { if (htim->Instance == TIM1) HAL_IncTick(); }
void Error_Handler(void) { __disable_irq(); while (1); }
