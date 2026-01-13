# PixRec

This repository contains the code to reproduce the experiments of the paper "PixRec: Leveraging Visual Context for Next-Item Prediction in Sequential Recommendation".

## Installation

To create a virtual environment before installing, you can use the command:
```bash
conda create -n rec_env python=3.11
conda activate rec_env
pip install -r requirements.txt
```

## Dataset
Download the datasets using the following commands:
```bash
# Create data directory (if directory doesn't exist)
mkdir data

# Download datasets
python dataset.py
```

## Experiments

### Training

To run the training and evaluation of baseline model (SmolLM2) on Amazon datasets, use the follow command:
```bash
python baseline.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct
```


To run the training and evaluation of PixRec Smol model(SmolVLM) without PEFT on Amazon dataset, use the follow command:
```bash
python amaz_train.py --model_name HuggingFaceTB/SmolVLM-256M-Instruct
```

To run the training and evaluation of PixRec Smol model(SmolVLM) or Paligemma2 with PEFT on Amazon dataset, use the follow command:
```bash
python amaz_train_peft.py --model_name HuggingFaceTB/SmolVLM-256M-Instruct --peft true
python amaz_train_peft.py --model_name google/paligemma2-3b-mix-224 --peft true
```

### Evaluation
Additional evaluation can be done by passing the mode as `test` similar to the following command:
```bash
python amaz_train.py --model_name HuggingFaceTB/SmolVLM-256M-Instruct --mode test
```

## Citation

If you use this codebase in academic work, please cite:

```
@misc{chakrabarty2026pixrecleveragingvisualcontext,
      title={PixRec: Leveraging Visual Context for Next-Item Prediction in Sequential Recommendation}, 
      author={Sayak Chakrabarty and Souradip Pal},
      year={2026},
      eprint={2601.06458},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2601.06458}, 
}
```

---

## License

Read the [LICENSE](LICENSE) file.

