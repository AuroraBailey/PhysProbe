# Scene annotation workflow

Scene audits provide sampling metadata: scene category, lighting descriptors,
and plausibility judgments. They are not physical-map targets or model inputs
for the official generation protocol.

The complete script roles, commands, file layout, and merge rules are documented
in [annotation/README.md](../annotation/README.md).

1. Place the intended views under one directory per scene.
2. Run the Doubao, GLM, and Qwen initial audits into separate directories.
3. Merge the three records using the offline merger.
4. Supply selected scenes to the reviewer when a second-stage review is needed.
   The current reviewer processes all matching scenes supplied to it; it does
   not select scenes by a review threshold.

A bundled [four-view scene-audit example](../examples/annotations/openroomff_mainaxis/README.md)
can be merged offline. It has no second-stage review because it did not meet
the original review-selection criteria.
