# CLASS COLLAPSE PROJECT

*Antoine Poirier*, *Florent Pollet*

[Class project](https://www.jeanfeydy.com/Teaching/index.html) based on the study of [this paper](https://www.mdpi.com/2813-0324/3/1/4).

**The report can be found [here](https://drive.google.com/file/d/1ce7IDspPz6T-n4BJ8ze9sglegq1TEYSB/view?usp=sharing).** The trained models for the report can be downloaded [here](https://drive.google.com/file/d/1BPWWO2yACTlUelyizWTewFyiEItB-QoC/view?usp=sharing).

## Installation

- Make sure you have Python installed (version 3.8 or above).
- Navigate to this folder.
- Run `pip install -e .[cuda] -U` or `pip install -e .[cpu] -U` (tested on Windows and Mac).

## Use

### Class collapse highlight on circle (section 3 of the report)

To run the script about the class circle study, you can just run the file `scripts/cc_circle.py` with Python.

### Using $L_{spread}$ on datasets (section 4 and 5 of the report)

To run the study on embeddings, you can use the console script `ccrun`, like `ccrun +dataset=house +loss=mse`.

You can choose a dataset among `house` and `synthetic`. You can choose a loss between `mse`, `supcon`, `nce`, `spread`.

Please feel free to tune other parameters by overriding them (please see `class_collapse/config` and Hydra documentation).

The output will be in the folder `outputs`, automatically generated.

## Troubleshooting

Please feel free to submit an issue if you have any questions.
