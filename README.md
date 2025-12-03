# DAIR-Music-Genre-Transfer
Exploring the use of machine learning and music information retrieval for instrument-specific symbolic music genre transfer 

## Repository Structure
```text
├── model/                    # VAE, LSTM encoder/decoder, adversarial classifier
├── dataset/                  # Data loading, piano roll processing
├── losses.py                 # VAE and adversarial loss functions
├── main.py                   # Training script
├── remi_eval.py              # Inference / evaluation script
├── opts.py                   # Argument parsing and experiment configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

## My Contributions
### Model Architecture
- Implemented the **AdversarialClassifier** (`nn.Module`) used for genre discrimination in the latent space.
- Developed the **EncoderLSTM** and **DecoderLSTM** modules, including bidirectional LSTMs, hidden-state reshaping, and linear projections for μ and σ.
- Integrated LSTM modules into a **modified VAE architecture**, enabling sequence-based latent encoding and reconstruction.

### Training & Loss Integration
- Added custom loss logic (`A_loss`) and wired the adversarial classifier into the VAE training loop.
- Modified latent sampling, forward passes, and data reshaping to support LSTM-based models.
- Updated argument parsing (`opts.py`) to support new hyperparameters, model types, and dataset paths.

### Data & Inference Pipeline
- Updated dataset handling to reshape and prepare symbolic music tensors for LSTM processing.
- Implemented a **full inference script**:
  - MIDI tokenization using REMI  
  - Sequence padding/truncation  
  - Running the trained model  
  - Genre transfer by manipulating latent vectors  
  - Converting output tokens back into MIDI  
