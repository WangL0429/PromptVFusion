# Upstream credits

The included backbone derives from [MAVFusion](https://github.com/lixilai/MAVFusion), *Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction*, by Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, and Haishu Tan. See [the paper](https://arxiv.org/abs/2604.01958). The tested causal feature interface extends that backbone. Its original acknowledgments identify engineering scaffolding from UniVF by Zixiang Zhao et al.

Optical flow uses [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) by Princeton Vision & Learning Lab. The flow wrapper and components retain their upstream names.

The original MAVFusion documentation links to [MAVFusion weights](https://drive.google.com/file/d/1KHRmjEUWVAKDQEZJdwIWA6ob32xF3__7/view) and [SEA-RAFT weights](https://drive.google.com/file/d/1C-pSoCi4qJREyL-eaOkWBWxsANAe3ZuS/view). Match the flow checkpoint filename in `config/module/spring-S.json`.
