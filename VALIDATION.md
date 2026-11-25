# Native validation

The two inference arms use the same epoch-2 motion-guided checkpoint, real-input masks, frame identities and color restoration. The candidate arm runs coherent foreground transport and background substitution before learned feature correction. The reference is MAVFusion with complete same-frame inputs and shared observed motion. Errors measure agreement with this reference.

Each of the 14 independent sources supplies two clips. Each clip starts with ten complete frames followed by seven frames missing one modality and seven frames missing the opposite modality. The initial missing modality differs between the two clips. This yields 392 missing frames at 640×480. The fixed showcase is independent of these 14 sources.

| After switching to missing VIS | Old-location edge error reduction | Current-location edge error reduction |
| --- | ---: | ---: |
| All seven frames | 1.20% | 1.40% |
| Seventh frame | 0.90% | 0.87% |

At the seventh frame, both edge errors decrease beyond a 1e-5 absolute tolerance in 4 of 14 sources and remain within that tolerance in 8. These are region averages. The first outage has no meaningful change because the coherent candidate path requires a newer real frame than the last synchronized anchor. The two additional traffic examples exhibit displacement and fragmented vehicles from the beginning of the evaluated VIS outage.
