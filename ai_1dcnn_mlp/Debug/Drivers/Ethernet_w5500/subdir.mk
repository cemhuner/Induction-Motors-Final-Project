################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/Ethernet_w5500/socket.c \
../Drivers/Ethernet_w5500/wizchip_conf.c 

OBJS += \
./Drivers/Ethernet_w5500/socket.o \
./Drivers/Ethernet_w5500/wizchip_conf.o 

C_DEPS += \
./Drivers/Ethernet_w5500/socket.d \
./Drivers/Ethernet_w5500/wizchip_conf.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/Ethernet_w5500/%.o Drivers/Ethernet_w5500/%.su Drivers/Ethernet_w5500/%.cyclo: ../Drivers/Ethernet_w5500/%.c Drivers/Ethernet_w5500/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F411xE -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/ST/AI/Inc -I../X-CUBE-AI/App -I../Middlewares/Third_Party/FreeRTOS/Source/include -I../Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS -I../Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM4F -I../Drivers/Ethernet_w5500/W5500 -I../Drivers/Ethernet_w5500 -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-Ethernet_w5500

clean-Drivers-2f-Ethernet_w5500:
	-$(RM) ./Drivers/Ethernet_w5500/socket.cyclo ./Drivers/Ethernet_w5500/socket.d ./Drivers/Ethernet_w5500/socket.o ./Drivers/Ethernet_w5500/socket.su ./Drivers/Ethernet_w5500/wizchip_conf.cyclo ./Drivers/Ethernet_w5500/wizchip_conf.d ./Drivers/Ethernet_w5500/wizchip_conf.o ./Drivers/Ethernet_w5500/wizchip_conf.su

.PHONY: clean-Drivers-2f-Ethernet_w5500

