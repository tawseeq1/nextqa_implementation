import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MiniCPMV(nn.Module):
    def __init__(self, video_feature_dim, hidden_dim, vocab_size, use_bert):
        super(MiniCPMV, self).__init__()
        self.use_bert = use_bert
        self.hidden_dim = hidden_dim

        self.video_encoder = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
        self.video_encoder.eval()

        self.question_encoder = AutoModel.from_pretrained('bert-base-uncased') if use_bert else nn.Embedding(vocab_size, hidden_dim)

        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, video_features, questions, question_lengths=None):
        video_embeds = self.video_encoder(video_features).last_hidden_state.mean(dim=1)

        if self.use_bert:
            question_embeds = self.question_encoder(questions)[0][:, 0, :] 
        else:
            question_embeds = self.question_encoder(questions).mean(dim=1)

        combined = video_embeds + question_embeds

        logits = self.fc(combined)
        return logits
    
def load_model(vocab, use_bert=True, hidden_dim=256, video_feature_dim=512):
    """
    Load the MiniCPMV model with the appropriate settings.
    :param vocab: Vocabulary object
    :param use_bert: Whether to use BERT
    :param hidden_dim: Hidden size for embeddings
    :param video_feature_dim: Dimensionality of the video features
    :return: Model instance
    """
    vocab_size = len(vocab)
    model = MiniCPMV(video_feature_dim, hidden_dim, vocab_size, use_bert)
    model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return model