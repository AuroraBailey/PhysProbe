# Data Sources and Attribution

PhysProbe uses three sources with target-specific eligibility. The manifests
identify the retained samples; they do not contain the source images or dense
ground-truth maps. Download data from the providers below and resolve manifest
paths against your local source/split roots as described in the
[data index](README.md).

| Source | Manifest `datasource` | Targets evaluated in this benchmark | Official access |
| --- | --- | --- | --- |
| OpenRooms-FF | `openroomff` | Depth, normal, albedo, roughness | [Official repository](https://github.com/bring728/OpenRooms_FF) |
| InteriorVerse | `interiorverse` | Depth, normal, albedo | [Dataset website](https://interiorverse.github.io/) · [Official repository](https://github.com/jingsenzhu/IndoorInverseRendering) |
| PBR-Rooms | `PBR-Rooms` | Metallic | [Hugging Face dataset](https://huggingface.co/datasets/pbr-rooms-benchmark/PBR-Rooms/tree/main) |

Eligibility describes this benchmark's scoring protocol, not the complete channel
coverage of each source. In particular, InteriorVerse roughness/metallic channels
are excluded by the paper's channel-reliability screening; PBR-Rooms exports
additional channels but is used here for metallic evaluation.

## OpenRooms-FF

OpenRooms-FF extends [OpenRooms](https://github.com/ViLab-UCSD/OpenRooms) with
forward-facing view groups. Request full access using the
[provider's form](https://github.com/bring728/OpenRooms_FF/blob/main/Request_for_Dataset.pdf)
and send the completed form to `jhcho@kist.re.kr`, following the official
repository instructions.

The provider publishes an
[MIT license](https://github.com/bring728/OpenRooms_FF/blob/main/LICENSE.md).
Its request form also asks recipients to notify the provider before publicly
releasing a modified or improved dataset, and lists upstream asset sources.
Retain the provider's notices and consult those access conditions when
redistributing data.

For attribution, cite [MAIR: Multi-view Attention Inverse Rendering with 3D
Spatially-Varying Lighting Estimation](https://arxiv.org/abs/2303.12368) and the
original [OpenRooms paper](https://openaccess.thecvf.com/content/CVPR2021/html/Li_OpenRooms_An_Open_Framework_for_Photorealistic_Indoor_Scene_Datasets_CVPR_2021_paper.html).
The published MAIR and OpenRooms entries are included in [references.bib](references.bib).
Use the benchmark manifests,
rather than the MAIR training/test lists, to select PhysProbe scoring samples.

## InteriorVerse

Obtain InteriorVerse through its [official website](https://interiorverse.github.io/).
The [official code and dataset repository](https://github.com/jingsenzhu/IndoorInverseRendering)
provides updated download links and recovery notices. Its 2026 updates supersede
the older email-request instructions on the dataset website; consult the current
repository before downloading. The provider reports that the original splits
are no longer valid after data recovery. PhysProbe sample selection remains
defined by this repository's retained manifests.
Read the [Terms of Use](https://interiorverse.github.io/files/Terms_of_Use.pdf)
when obtaining or using the data.

The terms specify **CC BY-NC-ND 4.0** and noncommercial research/education use.
Attribution alone does not remove the noncommercial or no-derivatives conditions.
Redistribution, including any processed dataset release, must respect the terms
and applicable rights in the scene assets.

Cite [Learning-Based Inverse Rendering of Complex Indoor Scenes with
Differentiable Monte Carlo Raytracing](https://doi.org/10.1145/3550469.3555407)
(SIGGRAPH Asia 2022), as requested by the provider. The provider's BibTeX is
included under `zhu2022learning` in [references.bib](references.bib).

## PBR-Rooms

PBR-Rooms is the companion dataset introduced with PhysProbe. Its release entry is:

[pbr-rooms-benchmark/PBR-Rooms on Hugging Face](https://huggingface.co/datasets/pbr-rooms-benchmark/PBR-Rooms/tree/main).

It is generated using an [Infinigen-based](https://github.com/princeton-vl/infinigen)
indoor rendering pipeline. The accompanying benchmark paper describes 1,000 scenes
and 39,000 RGB images with aligned physical channels. Metallic ground truth is
the renderer-exported PBR material parameter, rather than an estimate from RGB.
The metallic manifests select 960 main-axis images and 450 stress images.

PBR-Rooms is distributed under **CC BY-NC 4.0**. Consult its
[Hugging Face dataset card](https://huggingface.co/datasets/pbr-rooms-benchmark/PBR-Rooms)
and license files for the distributed assets and access instructions.
Attribute PBR-Rooms to the accompanying PhysProbe paper and acknowledge Infinigen
when using the underlying generation pipeline. Upstream software and assets
retain their own licenses.

## Citation guide

Downloadable entries are collected in [references.bib](references.bib).
Published-paper metadata comes from the CVF proceedings for MAIR/OpenRooms,
the InteriorVerse provider's BibTeX, and the official Infinigen repository.

| Resource used | References | BibTeX keys |
| --- | --- | --- |
| OpenRooms-FF images, ground truth, or bundled sample | [MAIR (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Choi_MAIR_Multi-View_Attention_Inverse_Rendering_With_3D_Spatially-Varying_Lighting_Estimation_CVPR_2023_paper.html) and [OpenRooms (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Li_OpenRooms_An_Open_Framework_for_Photorealistic_Indoor_Scene_Datasets_CVPR_2021_paper.html) | `choi2023mair`, `li2021openrooms` |
| InteriorVerse images or ground truth | [Learning-Based Inverse Rendering (SIGGRAPH Asia 2022)](https://doi.org/10.1145/3550469.3555407) | `zhu2022learning` |
| PBR-Rooms data or bundled sample | [PBR-Rooms dataset release](https://huggingface.co/datasets/pbr-rooms-benchmark/PBR-Rooms) and the accompanying PhysProbe paper | `pbrrooms` (dataset artifact only) |
| Infinigen generation pipeline | [Infinigen (CVPR 2023)](https://github.com/princeton-vl/infinigen#papers) and [Infinigen Indoors (CVPR 2024)](https://github.com/princeton-vl/infinigen#papers) | `infinigen2023infinite`, `infinigen2024indoors` |

The [PBR-Rooms citation section](https://huggingface.co/datasets/pbr-rooms-benchmark/PBR-Rooms#citation)
currently identifies an anonymous review artifact and defers a formal citation
until publication. The `pbrrooms` entry is a minimal dataset URL reference
prepared for this repository, not an official paper citation. It deliberately
omits unpublished author and publication metadata. Use the accompanying
PhysProbe manuscript as the benchmark reference during review; use the final
paper citation once it becomes available.

Cite the original data sources in addition to PhysProbe when using their samples.
Citation does not replace the source licenses or redistribution conditions.

## Calibration and derived assets

The [normal calibration manifests](manifests/calibration/README.md) reference
OpenRooms-FF and InteriorVerse samples. Selecting or exporting these samples
does not replace the original data licenses.

Roughness [segmentation maps](metadata/segmentation/README.md) are derived from
query RGB images using the [SAM3 workflow](../tools/segmentation/README.md).
They are auxiliary region maps, separate from source ground truth and valid
evaluation masks. SAM3 weights are obtained separately under the model provider's
terms; the bundled [SAM3 license notice](../tools/segmentation/SAM3_LICENSE)
documents the upstream dependency. The full segmentation collection is not bundled; the runnable
example includes one frozen region map. Follow the source-image terms when distributing derived assets.

## Scope of the repository license

The repository's [MIT license](../LICENSE) covers its original code. It does not
relicense third-party datasets, scene assets, model weights, or separately
licensed PBR-Rooms data. Keep provider license notices and paper citations with
any permitted redistribution. This page identifies sources and access terms;
the providers' license texts and agreements govern their data.
