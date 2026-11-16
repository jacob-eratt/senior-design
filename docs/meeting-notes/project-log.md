# RESNET8 Project Log

## 09/28 - 10/5
Project Progress
- Transitioned project to implementing RESNET8 accelerator
- Selected KV260 Kria Zynq FPGA for implementation (physical, not cloud)
- Created initial AI assisted design flows (manual, claudecode, chipagents)

Meeting Takeaways
- Set up repository for project files and code
- Get familiar with accelerator architectures
- Research potential alternatives

## 10/5 - 10/12
Project Progress
- Researched and learned functionality and layers of RESNET8
- Set up source code model for RESNET8 to implement into RTL
- Working FP32 and INT8 quantized software references and pretrained weights. 

Meeting Takeaways
- Access to ChipAgents, similar to ClaudeCode but with additional design specific tools
- Break down design into blocks with architectural spec per block (preferably .md files)

## 10/12 - 10/19
Project Progress
- Completed project proposal report outlining project objectives, plan, and timeline
- Started manual flow which will be most time intensive: get a working accelerator on FPGA
- Set up MAC unit design and initial ChatGPT/Codex generated RTL for Conv0 layer
- Initial breakdown of Conv0 functionality and architecture

Meeting Takeaways
- Start working on understanding and getting specific architecture RTL
- Split into sub-teams to work on different parts of the system

## 10/19 - 10/26
Project Progress
- Finalized Conv0 implementation INT8 types, MAC unit architecture
- Finalized global broadcast architecture and generated RTL (832 MACs, 32 x 27)
- Global broadcast had high resource usage (~98% CLB usage for Conv0 excluding interconnect)
- Proposed alternative systolic array and TPU-style architectures

Meeting Takeaways
- Perform risk-reduction of high resource usage by evaluating architecture alternatives
- Document AI usage for each flow, including the ones used for the current manual flow

## 10/26 - 11/2
Project Progress
- Set up new design for Conv0: weight stationary TPU-style architecture with less resource usage, pipelined inputs, and reused weights
- Had rough draft architectural documentation to generate RTL using Codex

Meeting Takeaways
- Continue exploring and implementing weight stationary architecture
- Set up presentation to show current progress on project implementation and caveats

## 11/2 - 11/9
Project Progress
- Finished RTL design generation of weight stationary architecture (702 PEs, 26 x 27, doubled buffered weights, partial sum outputs)
- Synthesis results for weight statioanry RTL: ~56% DSP, ~3% LUTs, ~7% FFs (no routing yet)
- Researched into KV260 SoC interconnect for transferring data between CPU and FPGA
- Started implementation of AXI DMA interfacing for loading inputs/weights into FPGA

Meeting Takeaways
- Continue interfacing tasks and get model running on FPGA
- Revisit AI assisted design flows to set up ClaudeCode and ChipAgents flows after manual flow is completed

## 11/9 - 11/16
Project Progress
Meeting Takeaways