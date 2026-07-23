# Exploratory Taxonomy-Light Metrics

| metric | method | dataset | split | value | delta_vs_baseline | status |
| --- | --- | --- | --- | --- | --- | --- |
| synonym_alias_collapsed_mIoU | sclip | voc20 | alias_collision_groups | 0.5038 | 0.0000 | no_duplicate_aliases_in_vocab |
| synonym_alias_collapsed_mIoU | corrclip | voc20 | alias_collision_groups | 0.6894 | 0.0000 | no_duplicate_aliases_in_vocab |
| thing_stuff_mIoU | sclip | coco_stuff164k | thing | 0.1270 |  | computed_from_coco_vocab_order |
| thing_stuff_mIoU | sclip | coco_stuff164k | stuff | 0.1347 |  | computed_from_coco_vocab_order |
| synonym_alias_collapsed_mIoU | sclip | coco_stuff164k | alias_collision_groups | 0.1312 | 0.0000 | no_duplicate_aliases_in_vocab |
| thing_stuff_mIoU | corrclip | coco_stuff164k | thing | 0.3018 |  | computed_from_coco_vocab_order |
| thing_stuff_mIoU | corrclip | coco_stuff164k | stuff | 0.1734 |  | computed_from_coco_vocab_order |
| synonym_alias_collapsed_mIoU | corrclip | coco_stuff164k | alias_collision_groups | 0.2232 | 0.0000 | no_duplicate_aliases_in_vocab |
| synonym_alias_collapsed_mIoU | sclip | ade847 | alias_collision_groups | 0.0130 | 0.0014 | computed_alias_collision_collapse |
| synonym_alias_collapsed_mIoU | corrclip | ade847 | alias_collision_groups | 0.0298 | 0.0008 | computed_alias_collision_collapse |
