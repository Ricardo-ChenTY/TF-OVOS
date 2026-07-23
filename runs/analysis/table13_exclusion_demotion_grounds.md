# Table 13 Exclusion Or Demotion Grounds

| Method | Task training? | Prompt/adapt. training? | MLLM call? | API-VLM? | Main reason |
| --- | --- | --- | --- | --- | --- |
| OVSeg | Yes | Yes | No | No | Kept as compact trained reference only. |
| SAN | Yes | Yes | No | No | Kept as compact trained reference only. |
| ODISE | Yes | Yes | No | No | Kept as compact trained reference only. |
| OVCoser | Yes | Yes | No | No | OVCOS-specific; moved to hard-domain related work / appendix. |
| SuCLIP | Yes | Yes | No | No | OVCOS-specific; moved to hard-domain related work / appendix. |
| COCUS / cascaded VLM OVCOS | Yes | Yes | No | No | Related hard-domain work; not part of general OVOS main rows. |
| DSS / Discover-Segment-Select | No | No | Yes | No | Uses inference-time MLLM mask selection; related work only. |
| GenSAM / ProMaC | No | No | Yes | No | Inference-time generative MLLM; related work only. |
| API-VLM + SAM | No | No | Depends | Yes | Closed-source API; not reproducible under strict protocol. |
