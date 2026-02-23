This project combines the domains of induction motors, embedded systems, and deep learning.

The primary aim of this project is to detect induction motor faults by running a deep learning model on an embedded device. Considering the energy efficiency and cost-effectiveness of embedded devices, this proposed solution is highly advantageous for environments where massive SCADA systems are impractical or too expensive to install.
Although the project focuses on detecting malfunctions in induction motors, the broader goal is to combine data-driven engineering with artificial intelligence and adapt it for embedded systems. From a wider perspective, this project serves as a framework for fault detection and predictive maintenance in standalone systems. Considering these features, the proposed method can be easily applied to individual devices with limited power capacities, such as satellites and rovers.

This project aims to accomplish two main milestones:

1. Detecting induction motor faults (Done)

2. Running a deep learning model on an embedded device (Done)

By achieving these two key concepts, almost 90% of the project is complete. However, the base structure, RTOS tasks, data transferring, buffering, and several other components still need to be reviewed.
Currently, the fault detection inference time does not meet the project requirements. It is running much slower than expected (approximately 0.125 detections/second). Semaphore/mutex optimization and deep learning model quantization/optimization are expected to resolve most of these performance bottlenecks.

Major problems encountered during development:

1. Incompatible Libraries: In STM32CUBEIDE version 1.3.1, the DSP Library does not perfectly match the HAL Library. You have to add their directories and files manually. Fortunately, I had installed the files beforehand, but as you can imagine, identifying this issue took a significant amount of time (approx. one week).
2. DMA Linkage Failure: In STM32CUBEIDE version 1.3.1, even if you link the DMA request to SPI communication in CubeMX, the code generation tool sometimes fails to link them in the definition functions. Because of this, the DMA is never actually called in the code. This requires careful debugging and manual linkage to fix.
3. Corrupted SPI DMA ISR: Also in version 1.3.1, the SPI DMA ISR might generate corrupted or blank. This happened once during development. The solution is to write the ISR manually.
4. RTOS Bottlenecks: RTOS Task priorities, queues, and stack memories are only functioning smoothly with very specific settings (as seen in the code). Intuitively, there is a bottleneck here that requires further debugging.
