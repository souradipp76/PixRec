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
@InProceedings{10.1007/978-3-032-24810-7_46,
      author="Chakrabarty, Sayak
      and Pal, Souradip",
      editor="Arai, Kohei
      and Lorenz, Pascal",
      title="PixRec: Leveraging Visual Context for Next-Item Prediction in Sequential Recommendation",
      booktitle="Intelligent Computing",
      year="2026",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="763--775",
      abstract="Large Language Models (LLMs) have recently shown strong potential for usage in sequential recommendation tasks through text-only models, which combine advanced prompt design, contrastive alignment, and fine-tuning on downstream domain-specific data. While effective, these approaches overlook the rich visual information present in many real-world recommendation scenarios, particularly in e-commerce. This study proposes PixRec - a vision-language framework that incorporates both textual attributes and product images into the recommendation pipeline. Our architecture leverages a vision--language model backbone capable of jointly processing image--text sequences, maintaining a dual-tower structure and mixed training objective while aligning multi-modal feature projections for both item--item and user--item interactions. Using the Amazon Reviews dataset augmented with product images, our experiments demonstrate {\$}{\$}3{\backslash}times {\$}{\$}3{\texttimes}and 40{\%} improvements in top-rank and top-10 rank accuracy over text-only recommenders, respectively, indicating that visual features can help distinguish items with similar textual descriptions. Our work outlines future directions for scaling multi-modal recommenders training, enhancing visual--text feature fusion, and evaluating inference-time performance. This work takes a step toward building software systems utilizing visual information in sequential recommendation for real-world applications like e-commerce.",
      isbn="978-3-032-24810-7"
}
```

---

## License

Read the [LICENSE](LICENSE) file.

