| Property           |                    Value |
| ------------------ | -----------------------: |
| Dataset            |                 Flickr8k |
| Unique Images      |                    8,091 |
| Caption Records    |                   40,455 |
| Captions per Image |                        5 |
| Task               |         Image Captioning |
| Input              |                    Image |
| Output             | Natural-language Caption |


# README 1 — ViT Model with ViTConfig

# Vision Transformer (ViT) from Scratch

## Overview

This project implements a Vision Transformer (ViT) architecture from scratch using PyTorch.

The primary objective of this project was to understand how Transformer architectures can be applied to computer vision and how an image can be converted into a sequence of tokens that can be processed using self-attention.

Instead of using a ready-made Vision Transformer implementation, the major components of the architecture were implemented manually, including:

- Image patch extraction
- Patch embeddings
- Learnable class token
- Positional embeddings
- Multi-Head Self-Attention
- Transformer Encoder blocks
- Layer Normalization
- Feed-Forward Networks
- Residual connections
- Dropout
- Classification head

The model architecture is controlled through a custom `ViTConfig` class, making the implementation configurable and easier to experiment with.

---

# Architecture

The overall Vision Transformer pipeline is:

```text
Input Image
     │
     ▼
Image Patching
     │
     ▼
Patch Embeddings
     │
     ▼
Add [CLS] Token
     │
     ▼
Add Positional Embeddings
     │
     ▼
Transformer Encoder
     │
     ├── Multi-Head Self-Attention
     ├── Layer Normalization
     ├── Feed-Forward Network
     └── Residual Connections
     │
     ▼
[CLS] Representation
     │
     ▼
Classification Head
     │
     ▼
Class Prediction
````

---

# Why Vision Transformer?

Traditional computer vision models such as CNNs process images primarily through convolutional operations.

Vision Transformers take a different approach.

The image is divided into fixed-size patches. Each patch is converted into an embedding, and the resulting embeddings are treated similarly to tokens in a natural-language Transformer.

For an image of size:

224 × 224 × 3

with a patch size of:
16 × 16

the image produces:


(224 / 16) × (224 / 16) = 14 × 14 = 196 patches

A learnable `[CLS]` token is then added, resulting in:

```text
196 + 1 = 197 tokens
```

These tokens are passed through the Transformer encoder.

---

# ViTConfig

A custom `ViTConfig` class is used to define the architecture.

This separates model configuration from model implementation.

Example configuration:

```python
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    hidden_size=768,
    num_heads=12,
    num_layers=12,
    intermediate_size=3072,
    num_classes=10,
    dropout=0.1
)
```

The configuration controls important architectural parameters such as:

* Input image size
* Patch size
* Number of input channels
* Transformer hidden dimension
* Number of attention heads
* Number of Transformer layers
* Feed-forward network dimension
* Number of output classes
* Dropout probability

This makes the model easier to modify without changing the underlying implementation.

---

# Main Components

## 1. Patch Embedding

The input image is divided into non-overlapping patches.

For a `224 × 224` image with `16 × 16` patches:

```text
224 × 224 image
        ↓
14 × 14 patch grid
        ↓
196 image patches
```

Each patch is then projected into the Transformer hidden dimension.

The result is a sequence of patch embeddings:

```text
[B, 196, hidden_size]
```

where `B` represents the batch size.

---

## 2. Class Token

A learnable `[CLS]` token is added to the beginning of the patch sequence.

The purpose of this token is to collect information from the complete image through the self-attention mechanism.

The resulting sequence becomes:

```text
[CLS] + Patch 1 + Patch 2 + ... + Patch 196
```

Therefore:

```text
[B, 197, hidden_size]
```

---

## 3. Positional Embeddings

Unlike CNNs, self-attention does not inherently know the spatial position of each image patch.

Learnable positional embeddings are therefore added to the patch embeddings.

This allows the model to distinguish between different spatial positions in the image.

---

# 4. Multi-Head Self-Attention

The core of the Vision Transformer is Multi-Head Self-Attention.

For each token, Query, Key and Value representations are generated.

The attention mechanism allows every patch to interact with every other patch.

Conceptually:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d) V
```

Using multiple attention heads allows the model to learn different types of relationships between image regions.

---

# 5. Transformer Encoder Block

Each Transformer encoder block consists of:

```text
Input
  │
  ▼
Layer Normalization
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Connection
```

The Feed-Forward Network expands the hidden representation before projecting it back to the original hidden dimension.

For the configuration used in this project:

```text
Hidden Size       = 768
Attention Heads   = 12
FFN Dimension     = 3072
```

---

# 6. Classification Head

After the final Transformer layer, the representation corresponding to the `[CLS]` token is extracted.

This representation is passed through a classification layer to produce the final class logits.

```text
Final Transformer Output
          │
          ▼
      [CLS] Token
          │
          ▼
   Linear Classification
          │
          ▼
      Class Logits
```

---

# Model Configuration

The main configuration used in the project is:

| Parameter             |     Value |
| --------------------- | --------: |
| Image Size            | 224 × 224 |
| Patch Size            |   16 × 16 |
| Number of Channels    |         3 |
| Number of Patches     |       196 |
| Sequence Length       |       197 |
| Hidden Size           |       768 |
| Attention Heads       |        12 |
| Transformer Layers    |        12 |
| FFN Dimension         |      3072 |
| Dropout               |       0.1 |
| Classification Token  |       Yes |
| Positional Embeddings | Learnable |

---

# Implementation

The project was implemented using PyTorch rather than directly using a pre-built ViT architecture.

Important modules implemented include:

```text
ViTConfig
PatchEmbedding
MultiHeadSelfAttention
TransformerEncoderBlock
TransformerEncoder
VisionTransformer
ClassificationHead
```

The configuration-driven design allows the architecture to be modified through `ViTConfig`.

---

# Training Pipeline

The training workflow follows:

```text
Dataset
   ↓
Image Preprocessing
   ↓
Patch Embedding
   ↓
Transformer Encoder
   ↓
CLS Token Representation
   ↓
Classification Head
   ↓
Loss Calculation
   ↓
Backpropagation
   ↓
Optimizer Update
```

The model was trained using a standard PyTorch training loop with separate training and validation stages.

---

# Evaluation

The model was evaluated on a held-out validation/test set using classification performance.

The main evaluation metric was:

```text
Accuracy
```

### Results

> Replace the values below with the exact final values from the ViT notebook.

```text
Training Accuracy : XX.XX%
Validation Accuracy : XX.XX%
Test Accuracy : XX.XX%
```

Training and validation loss curves were also monitored to understand model convergence and detect overfitting.

---

# Key Learnings

This project provided hands-on understanding of:

* How images can be converted into token sequences.
* How patch embeddings work.
* Why Vision Transformers require positional information.
* How Multi-Head Self-Attention operates on image patches.
* How `[CLS]` tokens are used for classification.
* How Transformer encoder blocks are constructed.
* How residual connections and LayerNorm stabilize Transformer training.
* How to build configurable deep-learning architectures.
* How Vision Transformers differ from conventional CNN architectures.

---

# Project Structure

vit-from-scratch/
│
├── notebook/
│   └── ViT_Implementation.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Jupyter Notebook
* Google Colab

---

# Future Improvements

Possible extensions include:

* Training on larger datasets.
* Adding stronger image augmentation.
* Experimenting with different patch sizes.
* Experimenting with different Transformer depths.
* Visualizing attention maps.
* Comparing ViT against CNN architectures.
* Loading and fine-tuning pretrained ViT weights.

---

# Conclusion

This project demonstrates a complete implementation of a Vision Transformer using PyTorch.

The main focus was not simply using ViT as a pretrained model, but understanding and implementing the individual components that make up the architecture.

The configurable `ViTConfig` design also provides a flexible foundation for building and experimenting with different Transformer-based vision architectures.

---

# README 2 — BLIP Image Captioning

This one is more directly connected to your **Computer Vision + Generative AI** profile.

```markdown
# BLIP — Vision-Language Image Captioning

## Overview

This project explores image captioning using BLIP (Bootstrapping Language-Image Pre-training), a Vision-Language Model designed to learn the relationship between visual information and natural language.

The objective is to generate a natural-language description of an image.

The project demonstrates the complete image-to-text pipeline:

```text
Image
  ↓
Image Preprocessing
  ↓
Vision Encoder
  ↓
Vision-Language Representation
  ↓
Language Model
  ↓
Text Generation
  ↓
Generated Caption
````

Given an image, the model analyzes its visual content and generates a textual description.

---

# What is Image Captioning?

Image captioning is a multimodal AI task where a model receives an image and generates a natural-language description of the visual content.

For example:

```text
Input:
An image containing a dog playing outside.

Output:
"A dog is playing outside."
```

The task requires the model to understand both:

1. Visual information
2. Natural language

This makes image captioning an important Vision-Language application.

---

# BLIP Architecture

The BLIP architecture combines a vision encoder with language modeling components.

The high-level architecture used in this project is:

```text
                  IMAGE
                    │
                    ▼
             Vision Encoder
                    │
                    ▼
            Visual Features
                    │
                    ▼
        Vision-Language Modeling
                    │
                    ▼
              Language Model
                    │
                    ▼
             Token Generation
                    │
                    ▼
             Generated Caption
```

The model learns to connect visual representations with natural-language descriptions.

---

# Main Components

## 1. Image Processing

Input images are converted into the format expected by the vision-language model.

The preprocessing pipeline includes:

* Image loading
* RGB conversion
* Resizing
* Tensor conversion
* Normalization

The processed image is then passed to the vision encoder.

---

# 2. Vision Encoder

The vision encoder converts the input image into a numerical representation.

Instead of directly generating text from raw pixels, the model first creates visual features representing information such as:

* Objects
* People
* Scenes
* Actions
* Visual relationships

These visual representations are then used by the language-generation component.

---

# 3. Vision-Language Representation

The visual features are combined with language information so that the model can condition language generation on the image.

This is the central idea behind Vision-Language Models:

```text
Visual Information + Language Modeling
                ↓
        Multimodal Understanding
```

---

# 4. Language Generation

The language model generates the caption token by token.

Conceptually:

```text
Image
 ↓
Visual Features
 ↓
Language Model
 ↓
"The"
 ↓
"The dog"
 ↓
"The dog is"
 ↓
"The dog is playing"
 ↓
"The dog is playing outside."
```

The final sequence of tokens is decoded into a human-readable caption.

---

# Dataset

The project uses an image-caption dataset containing images paired with natural-language descriptions.

The dataset follows the general structure:

```text
Image → Caption
```

Each image can have multiple human-written captions.

This allows the model's generated descriptions to be compared against reference descriptions.

---

# Training Pipeline

The training/evaluation workflow consists of:

```text
Load Dataset
      ↓
Load Image
      ↓
Preprocess Image
      ↓
Tokenize Caption
      ↓
Extract Visual Features
      ↓
Vision-Language Modeling
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Model Update
      ↓
Generate Captions
      ↓
Evaluate Captions
```

---

# Caption Generation

After training, the model can be used to generate captions for previously unseen images.

The inference process is:

```text
Input Image
      ↓
Image Preprocessing
      ↓
Vision Encoder
      ↓
Visual Representation
      ↓
Language Generation
      ↓
Generated Caption
```

The generated token IDs are decoded using the tokenizer to obtain the final caption.

---

# Evaluation Metrics

Generated captions were evaluated using commonly used image-captioning metrics.

The evaluation includes:

* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4
* METEOR
* ROUGE-L

These metrics measure different types of similarity between generated captions and reference captions.

# Qualitative Evaluation

In addition to numerical metrics, generated captions were manually inspected against the reference captions.

The qualitative evaluation helps determine whether the generated caption actually describes the image correctly, since automatic text-overlap metrics do not always capture semantic correctness.

For example:

```text
Reference:
"A dog is playing in the water."

Generated:
"A dog is playing in the water."
```

This represents strong visual and linguistic alignment.

Another example might use different wording while describing the same visual content:

```text
Reference:
"A child is playing with a dog."

Generated:
"A young boy plays with a dog."
```

Although the wording differs, the generated caption can still be semantically correct.

---

# Automatic Evaluation vs Qualitative Evaluation

Two types of evaluation were considered.

## Automatic Evaluation

Metrics such as BLEU, METEOR and ROUGE-L provide numerical measurements of similarity between generated and reference captions.

## Qualitative Evaluation

Manually inspecting generated captions helps determine:

* Whether the correct objects were identified.
* Whether actions were described correctly.
* Whether the scene was understood.
* Whether the generated sentence is grammatically meaningful.
* Whether the caption is semantically relevant to the image.

Both forms of evaluation are useful for image-captioning systems.

---

# Key Learning Outcomes

Through this project, I gained practical experience with:

* Vision-Language Models.
* Image captioning.
* Multimodal learning.
* Vision encoders.
* Transformer-based language generation.
* Image preprocessing.
* Text tokenization.
* Autoregressive text generation.
* Hugging Face Transformers.
* Evaluation of generated text.
* BLEU, METEOR and ROUGE metrics.
* Qualitative evaluation of generated captions.

---

# Applications

Image captioning can be used in:

* Accessibility tools for visually impaired users.
* Automatic image description.
* Image search systems.
* Digital asset management.
* Content moderation and organization.
* Assistive technologies.
* Social media image understanding.
* Multimodal AI systems.

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* BLIP
* Vision Transformers
* Natural Language Processing
* Computer Vision
* Google Colab

---

# Project Structure

```text
blip-image-captioning/
│
├── notebook/
│   └── BLIP_Image_Captioning.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# Future Improvements

Potential improvements include:

* Fine-tuning on larger image-caption datasets.
* Improving visual-language alignment.
* Evaluating against multiple reference captions.
* Adding more advanced captioning metrics.
* Extending the system toward Visual Question Answering.
* Extending the model toward conversational multimodal AI.
* Comparing BLIP with newer Vision-Language Models.

---

# Conclusion

This project demonstrates the use of a Vision-Language Model for converting visual information into natural-language descriptions.

The project provided practical experience in combining Computer Vision and Natural Language Processing and understanding how modern multimodal models connect image representations with language generation.

It also provided hands-on experience with both automatic evaluation metrics and qualitative analysis of generated captions.

---

For the **ViT and BLIP READMEs**, I don't want to guess the missing result numbers. We have exact results for your **current BLIP-2-inspired project**—including the 1,000-image setup, 800/200 image split, batch size 4, 3-epoch training decision, loss behavior, and the latest:

**BLEU-1 20.15%, BLEU-2 9.85%, BLEU-3 5.72%, BLEU-4 3.62%, METEOR 23.68%, ROUGE-L 23.84%.**



<img width="1920" height="1028" alt="image" src="https://github.com/user-attachments/assets/3e008a14-082e-4499-aa3c-0215fab76ae0" />



<img width="1920" height="1028" alt="image" src="https://github.com/user-attachments/assets/46a02f19-9e86-45ca-9751-5a3e63b7eb36" />



