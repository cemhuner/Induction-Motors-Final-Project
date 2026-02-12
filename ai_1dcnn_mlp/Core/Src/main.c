
// STM32F411: AI Node
// First Release: 01/02/2026 version: 0.1

#include "main.h"
#include "cmsis_os.h"
#include "wizchip_conf.h"
#include "socket.h"
#include <string.h>


#include "network.h"
#include "network_data.h"

SPI_HandleTypeDef hspi1;
DMA_HandleTypeDef hdma_spi1_rx;
DMA_HandleTypeDef hdma_spi1_tx;

osThreadId TaskRouterHandle;

osSemaphoreId spiMutex, dmaTxSem, dmaRxSem;

uint8_t destIP_PC[4]   = {192, 168, 1, 103};

wiz_NetInfo netInfoF411 = {
    .mac  = {0x00, 0x08, 0xDC, 0x11, 0x22, 0x33},
    .ip   = {192, 168, 1, 107},
    .sn   = {255, 255, 255, 0},
    .gw   = {192, 168, 1, 1},
    .dns  = {8, 8, 8, 8},
    .dhcp = NETINFO_STATIC
};

#define SOCK_RX_F401    0
#define SOCK_TX_PC      1

#define PORT_RX_F401    6000
#define PORT_TX_PC      6001

#define PKT_SIZE_FAST 404
#define PKT_SIZE_SLOW 644

// Temp buffer
uint8_t transfer_buffer[1024];

// --- AI global variables ---
// Assign as global variables to prevent stack overflow
ai_buffer ai_input[AI_NETWORK_IN_NUM];
ai_buffer ai_output[AI_NETWORK_OUT_NUM];
ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

ai_float in_fft[AI_NETWORK_IN_1_SIZE];      // 160 float
ai_float in_phys[AI_NETWORK_IN_2_SIZE];     // 400 float
ai_float out_data[AI_NETWORK_OUT_1_SIZE];   // 6 float
ai_handle network = AI_HANDLE_NULL;

// --- Function Prototypes ---
void SystemClock_Config(void);
static void MX_GPIO_Init(void); static void MX_DMA_Init(void); static void MX_SPI1_Init(void);
void StartTaskRouter(void const * argument);
uint32_t Run_AI_Inference(void);

void W5500_Select(void) { HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET); }
void W5500_Deselect(void) { HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); }
void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi) { osSemaphoreRelease(dmaTxSem); }
void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi) { osSemaphoreRelease(dmaRxSem); }
void W5500_ReadBuff_OS(uint8_t* buff, uint16_t len) { HAL_SPI_Receive_DMA(&hspi1, buff, len); osSemaphoreWait(dmaRxSem, osWaitForever); }
void W5500_WriteBuff_OS(uint8_t* buff, uint16_t len) { HAL_SPI_Transmit_DMA(&hspi1, buff, len); osSemaphoreWait(dmaTxSem, osWaitForever); }
uint8_t W5500_ReadByte(void) { uint8_t b; HAL_SPI_Receive(&hspi1, &b, 1, 10); return b; }
void W5500_WriteByte(uint8_t b) { HAL_SPI_Transmit(&hspi1, &b, 1, 10); }

// --- AI WRAPPER ---
uint32_t Run_AI_Inference(void) {
    if (network == AI_HANDLE_NULL) return 0;

    
    // Input-1 160 FFT
    ai_input[0].format = AI_NETWORK_IN_1_FORMAT;
    ai_input[0].data   = AI_HANDLE_PTR(in_fft);
    // ai_input[0].size   = AI_NETWORK_IN_1_SIZE; // Gerekirse açılabilir

    // Input-2 200 Torque + 200 Speed
    ai_input[1].format = AI_NETWORK_IN_2_FORMAT;
    ai_input[1].data   = AI_HANDLE_PTR(in_phys);
    
    // Outputs (6)
    ai_output[0].format = AI_NETWORK_OUT_1_FORMAT;
    ai_output[0].data   = AI_HANDLE_PTR(out_data);

    // Run
    if (ai_network_run(network, &ai_input[0], &ai_output[0]) != 1) return 0;

    // Argmax (Find the highest)
    float max_prob = -1.0f; int max_idx = 0;
    for(int i=0; i<AI_NETWORK_OUT_1_SIZE; i++) {
        if(((float*)out_data)[i] > max_prob) { max_prob = ((float*)out_data)[i]; max_idx = i; }
    }
    return (uint32_t)max_idx;
}

int main(void) {
    HAL_Init(); SystemClock_Config(); MX_GPIO_Init(); MX_DMA_Init(); MX_SPI1_Init();

    // Reset
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7, GPIO_PIN_RESET); HAL_Delay(10); HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7, GPIO_PIN_SET); HAL_Delay(100);

    reg_wizchip_cs_cbfunc(W5500_Select, W5500_Deselect); reg_wizchip_spi_cbfunc(W5500_ReadByte, W5500_WriteByte); reg_wizchip_spiburst_cbfunc(W5500_ReadBuff_OS, W5500_WriteBuff_OS);

    osSemaphoreDef(SEM_SPI); spiMutex = osSemaphoreCreate(osSemaphore(SEM_SPI), 1);
    osSemaphoreDef(SEM_DTX); dmaTxSem = osSemaphoreCreate(osSemaphore(SEM_DTX), 1); osSemaphoreWait(dmaTxSem, 0);
    osSemaphoreDef(SEM_DRX); dmaRxSem = osSemaphoreCreate(osSemaphore(SEM_DRX), 1); osSemaphoreWait(dmaRxSem, 0);

    // The only task
    osThreadDef(Router, StartTaskRouter, osPriorityNormal, 0, 2048);
    TaskRouterHandle = osThreadCreate(osThread(Router), NULL);

    osKernelStart(); while(1);
}

void StartTaskRouter(void const * argument) {
    osSemaphoreWait(spiMutex, osWaitForever);
    uint8_t memsize[2][8]={{4,4,2,2,2,2,0,0},{4,4,2,2,2,2,0,0}};
    ctlwizchip(CW_INIT_WIZCHIP,(void*)memsize);
    ctlnetwork(CN_SET_NETINFO,(void*)&netInfoF411);

    socket(SOCK_RX_F401, Sn_MR_TCP, PORT_RX_F401, 0); listen(SOCK_RX_F401);
    socket(SOCK_TX_PC, Sn_MR_TCP, 6002, 0);
    osSemaphoreRelease(spiMutex);

    // --- AI INIT ---
    ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    ai_network_params params = { .params = AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()), .activations = activations };
    ai_network_init(network, &params);
    // ------------------

    uint32_t last_conn_try = 0;
    uint8_t pc_connected = 0;
    
    // AI Flags
    int fast_cnt = 0;
    int has_fft = 0;
    uint32_t current_fault_code = 0;

    for(;;) {
        osSemaphoreWait(spiMutex, osWaitForever);

        // 1. PC connection
        uint8_t s_pc = getSn_SR(SOCK_TX_PC);
        if(s_pc == SOCK_ESTABLISHED) pc_connected = 1;
        else if(s_pc == SOCK_CLOSED || s_pc == SOCK_INIT) {
            pc_connected = 0;
            if(HAL_GetTick() - last_conn_try > 1000) {
                socket(SOCK_TX_PC, Sn_MR_TCP, 6002, 0);
                connect(SOCK_TX_PC, destIP_PC, PORT_TX_PC);
                last_conn_try = HAL_GetTick();
            }
        } else if(s_pc == SOCK_CLOSE_WAIT) { disconnect(SOCK_TX_PC); close(SOCK_TX_PC); pc_connected = 0; }

        // 2. Query
        uint8_t s_rx = getSn_SR(SOCK_RX_F401);
        if(s_rx == SOCK_ESTABLISHED) {
            uint16_t len = getSn_RX_RSR(SOCK_RX_F401);
            HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_8);

            if(len >= 4) { 
                recv(SOCK_RX_F401, transfer_buffer, 4);
                uint32_t header = *(uint32_t*)transfer_buffer;

                uint16_t payload_len = 0;
                
                // --- FAST DATA ---
                if(header == 0x1111AAAA) {
                    payload_len = PKT_SIZE_FAST - 4; // 400 Byte
                    recv(SOCK_RX_F401, transfer_buffer + 4, payload_len);

                    
                    if(fast_cnt < 4) {
                        
                        memcpy((uint8_t*)in_phys + (fast_cnt * 200), transfer_buffer + 4, 200);
                        
                        memcpy((uint8_t*)in_phys + 800 + (fast_cnt * 200), transfer_buffer + 4 + 200, 200);
                        fast_cnt++;
                    }

                    
                    if(pc_connected) {
                        send(SOCK_TX_PC, transfer_buffer, payload_len + 4);
                    }
                }
                // --- SLOW DATA (FFT) ---
                else if(header == 0x2222BBBB) {
                    payload_len = PKT_SIZE_SLOW - 4; // 640 Byte
                    recv(SOCK_RX_F401, transfer_buffer + 4, payload_len);
                    
                    // Copy for AI
                    memcpy(in_fft, transfer_buffer + 4, 640);
                    has_fft = 1;

                    // Run AI
                    if(fast_cnt >= 4 && has_fft) {
                        current_fault_code = Run_AI_Inference();
                        fast_cnt = 0; 
                        has_fft = 0;
                        
                        
                        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_8); // Debugging LED
                    }

                    // --- Final Packt ---
                    // Header(4) + Data(640) = 644th byte is the result
                    *(uint32_t*)(transfer_buffer + 644) = current_fault_code;

                    
                    if(pc_connected) {
                        send(SOCK_TX_PC, transfer_buffer, 648);
                    }
                }
                else {
                    // Header ERR shift one byte
                    recv(SOCK_RX_F401, transfer_buffer, 1);
                }
            }
        }
        else if(s_rx == SOCK_CLOSE_WAIT || s_rx == SOCK_CLOSED) {
            disconnect(SOCK_RX_F401); close(SOCK_RX_F401);
            socket(SOCK_RX_F401, Sn_MR_TCP, PORT_RX_F401, 0); listen(SOCK_RX_F401);
        }

        osSemaphoreRelease(spiMutex);
        osDelay(1);
    }
}

// ==================== HAL INIT ====================
void SystemClock_Config(void) {
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  HAL_RCC_OscConfig(&RCC_OscInitStruct);
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3);
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
    HAL_SPI_Init(&hspi1);

    hdma_spi1_rx.Instance = DMA2_Stream0;
    hdma_spi1_rx.Init.Channel = DMA_CHANNEL_3;
    hdma_spi1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_spi1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_spi1_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_spi1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_spi1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_spi1_rx.Init.Mode = DMA_NORMAL;
    hdma_spi1_rx.Init.Priority = DMA_PRIORITY_HIGH;
    HAL_DMA_Init(&hdma_spi1_rx);
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
    HAL_DMA_Init(&hdma_spi1_tx);
    __HAL_LINKDMA(&hspi1, hdmatx, hdma_spi1_tx);
}

static void MX_GPIO_Init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();

    // Reset Pin (GPIOC 7)
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7, GPIO_PIN_SET);
    GPIO_InitStruct.Pin = GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    // CS Pin (GPIOB 6)
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);
    GPIO_InitStruct.Pin = GPIO_PIN_6;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);
      GPIO_InitStruct.Pin = GPIO_PIN_8;
      GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
      GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
      HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

static void MX_DMA_Init(void) {
    __HAL_RCC_DMA2_CLK_ENABLE();
    HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
    HAL_NVIC_SetPriority(DMA2_Stream3_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream3_IRQn);
}

void StartDefaultTask(void const * argument) { for(;;) osDelay(1); }
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) { if (htim->Instance == TIM1) HAL_IncTick(); }
void Error_Handler(void) { __disable_irq(); while (1); }