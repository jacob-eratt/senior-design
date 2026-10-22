# Start here: ResNet-8 team catch-up

Start with the model and engineering explanation, then use the task cards to organize the working session.

1. Read or present [Understanding our ResNet-8 accelerator](MEETING_WALKTHROUGH.md), including questions.
2. Assign [working-session task cards](TASK_CARDS.md).
3. Use [the technical reference](TECHNICAL_REFERENCE.md) for details and evidence.

This folder organizes recent work without duplicating implementation files. The actual RTL, CPU code, fixtures, and reports stay in their existing locations.

**Baseline:** Conv0 passed RTL simulation and integrated routing/timing. Bitstream/XSA export succeeded. Vitis compilation and board execution have not been demonstrated in the reviewed evidence. Full ResNet exists in software; hardware currently computes Conv0 only.

| Meeting phase | Activity |
|---|---|
| Shared explanation | Model, operations, hardware decisions, evidence and questions |
| Assign work | Name owners/reviewers and confirm task inputs/tool access |
| Working groups | Deployment (3), agent experiment (2), verification/onboarding (2) |
| Review | Demonstrate results and cross-review |
| Closeout | Record blockers, next owners and check-in dates |

Seven attendees are assumed. With fewer people, preserve deployment and verification and reduce the experiment. Board PASS is a goal, not a promise within two hours.

The actual exported artifacts are under `.vivado/kv260_20261019_040420/` at repository root. Local `.vivado/` and `build/` paths may not exist on teammates' checkouts. Share matching artifacts with hashes and source provenance through the agreed team channel. The shared onboarding is maintained on the `resnet8-conv0` branch.
