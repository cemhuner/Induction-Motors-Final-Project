This project contains scientific contexts related to induction motors, embedded systems and deep learning.

The aim of this project, detecting induction motor faults through runnig a deep learning model on an embedded device.
Considering the energy efficiency and cost of embedded devices, it would be quite smart to use them in places where massive SCADA systems cannot be installed or are too expensive to install.
Although the project aims to troubleshoot malfunctions in induction motors, the real goal here is to combine data-driven engineering with artificial intelligence and adapt it to embedded systems.
The project can be viewed from a broader perspective, as fault detection and predictive maintenance in individual systems.
Considering the features listed above, the method proposed in this project can be easily applied to single devices with limited power capacities, such as SATELLITES and ROVERS.


This project is aiming to accomplish two milestone:
1-Detecting induction motor faults (done)
2-Running a deep learning model on an embedded device (done)

By achieving these two key concept, almost 90% of the project is done.
However, the base structure, RTOS tasks, data transferring, buffering and many more concepts must be reviewed.
The elapsed time that device detecting induction motor faults did not fulfill the project requirements so far. It is working way slower than expected. (Like 0.125 detection/second)
Semaphore and mutex optimization and deep learning model optimization is going to solve most of the performance problems (pure subjective opinion).


The list of some major encountered problems while developing this project:
1- In STM32CUBEIDE version 1.3.1, DSP Library is not matching with HAL Library. So you have to add their directories and files manually. 
   Thank God, I have been installed the files before this happened.
   As you can reckon, this took a lot of time to detect. (Approx. one week)
2- In STM32CUBEIDE version 1.3.1, despite you linked DMA request to SPI communication in CubeMX, the code generation tool won't link them in the definition functions.
   Due to this, DMA is never being called in the code while you were assuming it is called. If you are not careful enough, this took also a lot of time to detect. 
   So, you have to link them manually. 
3- In STM32CUBEIDE version 1.3.1, sometimes, SPI DMA ISR might be corrupted or blank. I don't know why but this happened just one time while developing. 
   The solution, again, write the ISR manually and it will be fixed.
4- RTOS Task priorities, queues and stack memories are working with just specific settings (as you might see in the code). 
   As an engineer, I can say that, purely intuitive, there is a bottleneck here.   
