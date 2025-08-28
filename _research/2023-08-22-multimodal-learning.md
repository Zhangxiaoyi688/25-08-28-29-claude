---
title: "Multimodal Learning for Visual Question Answering: Bridging Vision and Language Understanding"
authors: ["John Doe", "Sarah Wilson", "David Kim", "Emily Rodriguez"]
date: 2023-08-22
venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2023)"
type: "conference"
tags: ["multimodal", "computer-vision", "nlp", "visual-qa", "attention", "transformers"]
abstract: "We present a novel multimodal architecture that effectively combines visual and textual information for visual question answering tasks. Our approach introduces cross-modal attention mechanisms and hierarchical feature fusion to achieve state-of-the-art performance on VQA 2.0 and GQA datasets. The model demonstrates improved reasoning capabilities and better generalization to unseen question types, achieving 78.4% accuracy on VQA 2.0 test-std split."
pdf_url: "/assets/papers/multimodal-learning-2023.pdf"
external_url: "https://openaccess.thecvf.com/content/CVPR2023/html/Doe_Multimodal_Learning_CVPR_2023_paper.html"
doi: "10.1109/CVPR52729.2023.01234"
---

## Abstract

Visual Question Answering (VQA) represents a fundamental challenge in artificial intelligence, requiring systems to understand both visual content and natural language queries to provide accurate answers. Despite significant progress in recent years, existing approaches often struggle with complex reasoning tasks and fail to effectively integrate multimodal information.

In this work, we propose **Cross-Modal Reasoning Transformer (CMRT)**, a novel architecture that addresses these limitations through:

1. **Hierarchical Cross-Modal Attention**: Multi-level attention mechanisms that capture fine-grained visual-textual correspondences
2. **Adaptive Feature Fusion**: Dynamic integration of visual and linguistic representations
3. **Reasoning-Aware Training**: Curriculum learning strategy that progressively increases reasoning complexity

Our approach achieves state-of-the-art results on multiple VQA benchmarks, with particularly strong performance on questions requiring multi-step reasoning and compositional understanding.

## 1. Introduction

Visual Question Answering sits at the intersection of computer vision and natural language processing, requiring AI systems to demonstrate human-like understanding of visual scenes and linguistic queries. The task involves analyzing an image and answering questions about its content, relationships between objects, spatial reasoning, and even abstract concepts.

### 1.1 Challenges in VQA

Current VQA systems face several fundamental challenges:

**Multimodal Integration**: Effectively combining visual and textual information remains non-trivial. Simple concatenation or element-wise operations often fail to capture complex cross-modal relationships.

**Compositional Reasoning**: Many questions require understanding relationships between multiple objects, their attributes, and spatial configurations. For example: "What color is the shirt of the person standing next to the red car?"

**Language Bias**: VQA datasets often contain statistical biases that allow models to achieve reasonable performance without truly understanding visual content.

**Generalization**: Models trained on specific datasets often fail to generalize to new question types or visual domains.

### 1.2 Our Approach

We address these challenges through a unified architecture that:

- Employs hierarchical attention mechanisms to capture multi-scale visual-textual relationships
- Uses adaptive fusion strategies that dynamically weight different modalities based on question type
- Incorporates explicit reasoning modules that decompose complex questions into simpler sub-problems
- Leverages curriculum learning to improve generalization capabilities

## 2. Related Work

### 2.1 Early VQA Approaches

Early VQA systems typically followed a pipeline approach:

1. **Feature Extraction**: CNN features for images, RNN/LSTM for questions
2. **Fusion**: Simple concatenation or element-wise operations
3. **Classification**: Multi-layer perceptron for answer prediction

Notable examples include:
- **VQA baseline** [1]: Simple CNN + LSTM architecture
- **iBOWIMG** [2]: Bag-of-words question representation
- **DPPnet** [3]: Dynamic parameter prediction networks

### 2.2 Attention-Based Methods

The introduction of attention mechanisms significantly improved VQA performance:

**Visual Attention**:
- **SAN** [4]: Stacked attention networks for iterative reasoning
- **HieCoAtt** [5]: Hierarchical co-attention for question-image understanding

**Cross-Modal Attention**:
- **BAN** [6]: Bilinear attention networks
- **MUTAN** [7]: Multimodal tucker fusion

### 2.3 Transformer-Based Approaches

Recent work has adapted transformer architectures for VQA:

- **LXMERT** [8]: Cross-modal transformer with pre-training
- **UNITER** [9]: Universal image-text representation learning
- **ViLBERT** [10]: Vision-and-language BERT

### 2.4 Reasoning and Compositional Understanding

Several approaches focus on explicit reasoning:

- **MAC** [11]: Memory, attention, and composition networks
- **FiLM** [12]: Feature-wise linear modulation
- **Neural Module Networks** [13]: Compositional question decomposition

## 3. Method

### 3.1 Architecture Overview

Our Cross-Modal Reasoning Transformer (CMRT) consists of four main components:

1. **Visual Encoder**: Processes input images to extract hierarchical visual features
2. **Question Encoder**: Encodes natural language questions into contextual representations
3. **Cross-Modal Reasoning Module**: Performs iterative reasoning through attention mechanisms
4. **Answer Decoder**: Generates final answers based on fused representations

```python
class CMRT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_encoder = VisualEncoder(config)
        self.question_encoder = QuestionEncoder(config)
        self.reasoning_module = CrossModalReasoning(config)
        self.answer_decoder = AnswerDecoder(config)
    
    def forward(self, image, question, question_mask):
        # Extract visual features
        visual_features = self.visual_encoder(image)
        
        # Encode question
        question_features = self.question_encoder(question, question_mask)
        
        # Cross-modal reasoning
        fused_features = self.reasoning_module(
            visual_features, question_features, question_mask
        )
        
        # Generate answer
        answer_logits = self.answer_decoder(fused_features)
        
        return answer_logits
```

### 3.2 Visual Encoder

We employ a hierarchical visual encoder that captures multi-scale visual information:

#### 3.2.1 Backbone Network

We use a ResNet-152 backbone pre-trained on ImageNet, extracting features from multiple layers:

- **Low-level features** (conv2_x): Fine-grained details and textures
- **Mid-level features** (conv3_x, conv4_x): Object parts and local patterns
- **High-level features** (conv5_x): Semantic objects and global context

#### 3.2.2 Feature Pyramid Network

```python
class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = resnet152(pretrained=True)
        self.fpn = FeaturePyramidNetwork(
            in_channels=[256, 512, 1024, 2048],
            out_channels=config.visual_dim
        )
        self.spatial_encoder = SpatialEncoder(config)
    
    def forward(self, images):
        # Extract multi-scale features
        features = self.backbone.extract_features(images)
        
        # Feature pyramid network
        pyramid_features = self.fpn(features)
        
        # Add spatial encoding
        spatial_features = self.spatial_encoder(pyramid_features)
        
        return spatial_features
```

#### 3.2.3 Spatial Encoding

We incorporate explicit spatial information through learnable positional embeddings:

```python
class SpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_regions, config.visual_dim)
        )
    
    def forward(self, visual_features):
        # visual_features: [batch, regions, visual_dim]
        batch_size, num_regions, _ = visual_features.shape
        
        # Add positional embeddings
        pos_embed = self.pos_embed[:, :num_regions, :]
        spatial_features = visual_features + pos_embed
        
        return spatial_features
```

### 3.3 Question Encoder

The question encoder processes natural language queries using a transformer-based architecture:

```python
class QuestionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim
            ),
            num_layers=config.num_layers
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, question_tokens, attention_mask):
        # Token embeddings
        embeddings = self.embedding(question_tokens)
        
        # Transformer encoding
        question_features = self.transformer(
            embeddings.transpose(0, 1),
            src_key_padding_mask=~attention_mask
        ).transpose(0, 1)
        
        # Layer normalization
        question_features = self.layer_norm(question_features)
        
        return question_features
```

### 3.4 Cross-Modal Reasoning Module

The core innovation of our approach lies in the cross-modal reasoning module, which performs iterative reasoning through hierarchical attention mechanisms.

#### 3.4.1 Hierarchical Cross-Modal Attention

```python
class CrossModalReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_reasoning_steps = config.num_reasoning_steps
        
        # Multi-level attention layers
        self.visual_self_attention = MultiHeadAttention(config)
        self.question_self_attention = MultiHeadAttention(config)
        self.cross_attention = CrossModalAttention(config)
        
        # Fusion layers
        self.fusion_layer = AdaptiveFusion(config)
        
        # Memory mechanism
        self.memory = ReasoningMemory(config)
    
    def forward(self, visual_features, question_features, question_mask):
        # Initialize reasoning state
        reasoning_state = self.memory.initialize(
            visual_features, question_features
        )
        
        # Iterative reasoning
        for step in range(self.num_reasoning_steps):
            # Self-attention within modalities
            visual_attended = self.visual_self_attention(
                visual_features, visual_features, visual_features
            )
            
            question_attended = self.question_self_attention(
                question_features, question_features, question_features,
                key_padding_mask=~question_mask
            )
            
            # Cross-modal attention
            cross_attended = self.cross_attention(
                visual_attended, question_attended, reasoning_state
            )
            
            # Adaptive fusion
            fused_features = self.fusion_layer(
                visual_attended, question_attended, cross_attended
            )
            
            # Update reasoning state
            reasoning_state = self.memory.update(
                reasoning_state, fused_features, step
            )
        
        return reasoning_state
```

#### 3.4.2 Cross-Modal Attention Mechanism

```python
class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_proj = nn.Linear(config.visual_dim, config.hidden_dim)
        self.question_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
    def forward(self, visual_features, question_features, reasoning_state):
        # Project to common dimension
        visual_proj = self.visual_proj(visual_features)
        question_proj = self.question_proj(question_features)
        
        # Cross-modal attention: visual queries, question keys/values
        v2q_attended, v2q_weights = self.attention(
            visual_proj.transpose(0, 1),
            question_proj.transpose(0, 1),
            question_proj.transpose(0, 1)
        )
        
        # Cross-modal attention: question queries, visual keys/values
        q2v_attended, q2v_weights = self.attention(
            question_proj.transpose(0, 1),
            visual_proj.transpose(0, 1),
            visual_proj.transpose(0, 1)
        )
        
        # Combine attended features
        cross_attended = torch.cat([
            v2q_attended.transpose(0, 1),
            q2v_attended.transpose(0, 1)
        ], dim=-1)
        
        return cross_attended
```

#### 3.4.3 Adaptive Fusion

```python
class AdaptiveFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        self.fusion_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, visual_features, question_features, cross_features):
        # Compute fusion gates
        concat_features = torch.cat([
            visual_features.mean(dim=1),
            question_features.mean(dim=1),
            cross_features.mean(dim=1)
        ], dim=-1)
        
        gates = self.gate_network(concat_features)
        
        # Weighted fusion
        weighted_visual = gates[:, 0:1].unsqueeze(1) * visual_features
        weighted_question = gates[:, 1:2].unsqueeze(1) * question_features
        weighted_cross = gates[:, 2:3].unsqueeze(1) * cross_features
        
        # Final fusion
        fused = torch.cat([
            weighted_visual.mean(dim=1),
            weighted_question.mean(dim=1),
            weighted_cross.mean(dim=1)
        ], dim=-1)
        
        return self.fusion_network(fused)
```

### 3.5 Answer Decoder

The answer decoder generates final predictions based on the fused multimodal representations:

```python
class AnswerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_answers)
        )
        
    def forward(self, fused_features):
        return self.classifier(fused_features)
```

## 4. Training Strategy

### 4.1 Curriculum Learning

We employ a curriculum learning strategy that progressively increases the complexity of training examples:

```python
class CurriculumScheduler:
    def __init__(self, total_epochs, complexity_levels=4):
        self.total_epochs = total_epochs
        self.complexity_levels = complexity_levels
        self.current_level = 0
    
    def get_complexity_level(self, epoch):
        # Gradually increase complexity
        progress = epoch / self.total_epochs
        level = min(
            int(progress * self.complexity_levels),
            self.complexity_levels - 1
        )
        return level
    
    def filter_examples(self, dataset, complexity_level):
        # Filter examples based on question complexity
        filtered_examples = []
        for example in dataset:
            if example['complexity'] <= complexity_level:
                filtered_examples.append(example)
        return filtered_examples
```

### 4.2 Loss Function

We use a combination of cross-entropy loss and auxiliary losses:

```python
def compute_loss(predictions, targets, attention_weights, config):
    # Main classification loss
    ce_loss = F.cross_entropy(predictions, targets)
    
    # Attention regularization
    attention_reg = compute_attention_regularization(attention_weights)
    
    # Total loss
    total_loss = ce_loss + config.attention_reg_weight * attention_reg
    
    return total_loss

def compute_attention_regularization(attention_weights):
    # Encourage diverse attention patterns
    entropy_loss = -torch.sum(
        attention_weights * torch.log(attention_weights + 1e-8),
        dim=-1
    ).mean()
    
    return -entropy_loss  # Maximize entropy
```

### 4.3 Optimization

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing with warmup
- **Batch Size**: 128 (distributed across 8 GPUs)
- **Gradient Clipping**: Max norm of 1.0

## 5. Experiments

### 5.1 Datasets

We evaluate our approach on three standard VQA benchmarks:

**VQA 2.0** [14]:
- 1.1M questions on 200K images
- Balanced dataset to reduce language bias
- Test-dev and test-std splits for evaluation

**GQA** [15]:
- 22M questions on 113K images
- Compositional questions requiring multi-step reasoning
- Balanced training and evaluation splits

**VQA-CP v2** [16]:
- Rearranged VQA 2.0 to test generalization
- Different answer distributions in train/test
- Evaluates robustness to dataset bias

### 5.2 Implementation Details

**Model Configuration**:
- Visual encoder: ResNet-152 + FPN
- Hidden dimension: 768
- Number of attention heads: 12
- Number of reasoning steps: 3
- Dropout rate: 0.1

**Training Setup**:
- Epochs: 30 for VQA 2.0, 20 for GQA
- Learning rate: 1e-4 with cosine decay
- Warmup steps: 2000
- Weight decay: 1e-2

### 5.3 Baseline Comparisons

We compare against several state-of-the-art methods:

- **Bottom-Up Top-Down** [17]: Attention-based VQA
- **BAN** [6]: Bilinear attention networks
- **MCAN** [18]: Modular co-attention networks
- **LXMERT** [8]: Cross-modal transformer
- **UNITER** [9]: Universal image-text representation
- **VILLA** [19]: Vision-and-language pre-training

### 5.4 Main Results

#### 5.4.1 VQA 2.0 Results

| Model | Test-dev | Test-std |
|-------|----------|----------|
| Bottom-Up Top-Down | 65.32 | 65.67 |
| BAN | 70.04 | 70.35 |
| MCAN | 70.63 | 70.90 |
| LXMERT | 72.42 | 72.54 |
| UNITER | 73.82 | 74.02 |
| VILLA | 76.59 | 76.85 |
| **CMRT (Ours)** | **78.21** | **78.44** |

#### 5.4.2 GQA Results

| Model | Accuracy | Consistency | Validity | Plausibility |
|-------|----------|-------------|----------|-------------|
| Bottom-Up Top-Down | 49.74 | 78.71 | 96.18 | 84.57 |
| LXMERT | 60.00 | 89.52 | 99.24 | 90.45 |
| UNITER | 61.54 | 90.23 | 99.31 | 91.12 |
| **CMRT (Ours)** | **63.87** | **91.45** | **99.42** | **92.33** |

#### 5.4.3 VQA-CP v2 Results

| Model | Overall | Yes/No | Number | Other |
|-------|---------|--------|--------|---------|
| Bottom-Up Top-Down | 39.74 | 42.27 | 11.93 | 46.05 |
| LXMERT | 42.53 | 45.18 | 13.42 | 48.96 |
| **CMRT (Ours)** | **45.82** | **48.73** | **15.67** | **52.14** |

### 5.5 Ablation Studies

#### 5.5.1 Component Analysis

| Component | VQA 2.0 (test-dev) | GQA |
|-----------|---------------------|-----|
| Baseline (no cross-modal) | 71.23 | 58.45 |
| + Cross-modal attention | 75.67 | 61.32 |
| + Adaptive fusion | 77.12 | 62.78 |
| + Curriculum learning | **78.21** | **63.87** |

#### 5.5.2 Reasoning Steps Analysis

| Reasoning Steps | VQA 2.0 | GQA | Inference Time (ms) |
|-----------------|---------|-----|--------------------|
| 1 | 76.45 | 61.23 | 45 |
| 2 | 77.78 | 62.91 | 67 |
| 3 | **78.21** | **63.87** | 89 |
| 4 | 78.19 | 63.82 | 112 |
| 5 | 78.15 | 63.79 | 134 |

#### 5.5.3 Attention Visualization

![Attention Visualization](https://via.placeholder.com/800x400/1e3a8a/ffffff?text=Cross-Modal+Attention+Patterns)

*Figure 1: Visualization of cross-modal attention patterns. Our model learns to focus on relevant image regions based on question content.*

## 6. Analysis and Discussion

### 6.1 Question Type Analysis

We analyze performance across different question types:

**Counting Questions**: 15.2% improvement over LXMERT
- Better spatial reasoning through hierarchical attention
- Explicit object detection and counting mechanisms

**Spatial Reasoning**: 12.8% improvement
- Enhanced spatial encoding in visual features
- Cross-modal attention captures spatial relationships

**Attribute Recognition**: 8.4% improvement
- Fine-grained visual features from FPN
- Attribute-specific attention patterns

**Compositional Questions**: 18.7% improvement
- Multi-step reasoning through iterative attention
- Memory mechanism maintains intermediate results

### 6.2 Error Analysis

**Common Error Types**:

1. **Complex Spatial Reasoning** (23% of errors)
   - Questions involving multiple spatial relationships
   - "What is to the left of the object behind the car?"

2. **Fine-grained Recognition** (19% of errors)
   - Distinguishing between similar objects/attributes
   - "Is this a sedan or SUV?"

3. **Commonsense Reasoning** (16% of errors)
   - Questions requiring world knowledge
   - "Is it safe to cross the street?"

4. **Ambiguous Questions** (12% of errors)
   - Multiple valid interpretations
   - "What color is the shirt?" (multiple people present)

### 6.3 Computational Efficiency

**Training Efficiency**:
- 2.3x faster than LXMERT (due to efficient attention)
- 40% reduction in memory usage
- Scales linearly with sequence length

**Inference Speed**:
- 89ms per sample on V100 GPU
- Suitable for real-time applications
- Batch processing achieves 180 samples/second

### 6.4 Generalization Analysis

**Cross-Dataset Transfer**:
- Model trained on VQA 2.0 achieves 58.3% on GQA (zero-shot)
- Fine-tuning improves to 63.2% (vs. 63.87% trained from scratch)
- Demonstrates good transfer learning capabilities

**Domain Adaptation**:
- Medical VQA: 67.4% accuracy (fine-tuned from VQA 2.0)
- Scientific diagrams: 71.2% accuracy
- Shows promise for specialized domains

## 7. Limitations and Future Work

### 7.1 Current Limitations

**Computational Complexity**:
- Still requires significant computational resources
- Memory usage scales with image resolution
- Limited to single-image questions

**Reasoning Depth**:
- Struggles with very complex multi-hop reasoning
- Limited by fixed number of reasoning steps
- Difficulty with abstract conceptual questions

**Dataset Bias**:
- Performance still affected by dataset-specific biases
- Requires careful evaluation on out-of-distribution data
- Limited diversity in training examples

### 7.2 Future Directions

**Architectural Improvements**:
- Dynamic reasoning step allocation
- Integration with external knowledge bases
- Multi-image and video question answering

**Training Enhancements**:
- Self-supervised pre-training objectives
- Adversarial training for robustness
- Meta-learning for few-shot adaptation

**Applications**:
- Real-world deployment in assistive technologies
- Integration with robotics systems
- Educational and accessibility applications

## 8. Conclusion

We have presented the Cross-Modal Reasoning Transformer (CMRT), a novel architecture for visual question answering that effectively combines visual and textual information through hierarchical cross-modal attention mechanisms. Our approach achieves state-of-the-art performance on multiple VQA benchmarks while demonstrating improved reasoning capabilities and better generalization.

**Key Contributions**:

1. **Hierarchical Cross-Modal Attention**: Multi-level attention mechanisms that capture fine-grained visual-textual correspondences at different scales

2. **Adaptive Feature Fusion**: Dynamic integration strategy that learns to weight different modalities based on question requirements

3. **Reasoning-Aware Training**: Curriculum learning approach that progressively increases reasoning complexity during training

4. **Comprehensive Evaluation**: Extensive experiments demonstrating effectiveness across multiple datasets and question types

**Impact and Significance**:

- **Performance**: Achieves new state-of-the-art results on VQA 2.0 (78.44%) and GQA (63.87%)
- **Efficiency**: 2.3x faster training and reduced memory usage compared to existing methods
- **Generalization**: Strong performance on out-of-distribution evaluation (VQA-CP v2)
- **Interpretability**: Attention visualizations provide insights into model reasoning

**Broader Implications**:

Our work advances the field of multimodal AI by demonstrating how careful architectural design and training strategies can significantly improve visual question answering performance. The hierarchical attention mechanisms and adaptive fusion strategies introduced in CMRT are general techniques that could benefit other multimodal tasks such as image captioning, visual dialog, and multimodal machine translation.

The improved reasoning capabilities and generalization performance make our approach particularly suitable for real-world applications where robustness and reliability are crucial. As AI systems become more integrated into daily life, the ability to understand and reason about visual content in response to natural language queries will become increasingly important.

**Future Research Directions**:

While our work represents a significant step forward, several important challenges remain. Future research should focus on:

- Scaling to more complex reasoning tasks requiring deeper understanding
- Improving efficiency for deployment on resource-constrained devices
- Addressing dataset biases and improving fairness across different populations
- Extending to multi-image and video understanding tasks

We believe that continued progress in multimodal AI will require close collaboration between computer vision and natural language processing communities, along with careful consideration of ethical implications and real-world deployment challenges.

## Acknowledgments

We thank the anonymous reviewers for their constructive feedback and suggestions. We also acknowledge the computational resources provided by our institution's high-performance computing center. This work was supported by NSF grants IIS-2048280 and CCF-2112665, and a Google Research Award.

Special thanks to our colleagues who provided valuable discussions and insights throughout this project, and to the open-source community for making datasets and evaluation tools available.

## Code and Model Availability

To facilitate reproducibility and future research, we make our code, pre-trained models, and experimental results publicly available:

- **Code Repository**: https://github.com/johndoe/cmrt-vqa
- **Pre-trained Models**: https://huggingface.co/johndoe/cmrt-vqa
- **Experimental Data**: https://doi.org/10.5281/zenodo.7654321
- **Demo Interface**: https://cmrt-demo.example.com

## References

[1] Antol, S., et al. "VQA: Visual question answering." *ICCV* 2015.

[2] Zhou, B., et al. "Simple baseline for visual question answering." *arXiv:1512.02167* 2015.

[3] Noh, H., et al. "Image question answering using convolutional neural network with dynamic parameter prediction." *CVPR* 2016.

[4] Yang, Z., et al. "Stacked attention networks for image question answering." *CVPR* 2016.

[5] Lu, J., et al. "Hierarchical question-image co-attention for visual question answering." *NeurIPS* 2016.

[6] Kim, J.H., et al. "Bilinear attention networks." *NeurIPS* 2018.

[7] Ben-Younes, H., et al. "MUTAN: Multimodal tucker fusion for visual question answering." *ICCV* 2017.

[8] Tan, H., & Bansal, M. "LXMERT: Learning cross-modality encoder representations from transformers." *EMNLP* 2019.

[9] Chen, Y.C., et al. "UNITER: Universal image-text representation learning." *ECCV* 2020.

[10] Lu, J., et al. "ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks." *NeurIPS* 2019.

[11] Hudson, D.A., & Manning, C.D. "Compositional attention networks for machine reasoning." *ICLR* 2018.

[12] Perez, E., et al. "FiLM: Visual reasoning with a general conditioning layer." *AAAI* 2018.

[13] Andreas, J., et al. "Neural module networks." *CVPR* 2016.

[14] Goyal, Y., et al. "Making the V in VQA matter: Elevating the role of image understanding in visual question answering." *CVPR* 2017.

[15] Hudson, D.A., & Manning, C.D. "GQA: A new dataset for real-world visual reasoning and compositional question answering." *CVPR* 2019.

[16] Agrawal, A., et al. "Don't just assume; look and answer: Overcoming priors for visual question answering." *CVPR* 2018.

[17] Anderson, P., et al. "Bottom-up and top-down attention for image captioning and visual question answering." *CVPR* 2018.

[18] Yu, Z., et al. "Deep modular co-attention networks for visual question answering." *CVPR* 2019.

[19] Gan, Z., et al. "Large-scale adversarial training for vision-and-language representation learning." *NeurIPS* 2020.