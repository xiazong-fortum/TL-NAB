import torch
import torch.nn as nn

# class LSTMAutoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, dropout):
#         super(LSTMAutoencoder, self).__init__()
        
#         # Encoder
#         self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.layer_norm_lstm = nn.LayerNorm(hidden_dim)
#         self.encoder_fc = nn.Linear(hidden_dim, hidden_dim)
#         self.layer_norm_fc = nn.LayerNorm(hidden_dim)
#         self.encoder_dropout = nn.Dropout(dropout)
        
#         # Decoder
#         self.decoder_fc = nn.Linear(hidden_dim, hidden_dim)
#         self.decoder_dropout = nn.Dropout(dropout)
#         self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=dropout)

#     def forward(self, x):
#         # Encoder
#         _, (h, _) = self.encoder_lstm(x)
#         h = self.layer_norm_lstm(h)
#         encoded = self.encoder_dropout(self.encoder_fc(h[-1]))
#         encoded = self.layer_norm_fc(encoded)

#         # Decoder
#         decoder_input = self.decoder_dropout(self.decoder_fc(encoded)).unsqueeze(1).repeat(1, x.size(1), 1)
#         decoded, _ = self.decoder_lstm(decoder_input)
        
#         return decoded
    
import torch
import torch.nn as nn

class SimpleLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length, num_layers, dropout=0):
        super(SimpleLSTMAutoencoder, self).__init__()
        
        # Encoder: a single LSTM layer with dropout
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Decoder: a single LSTM layer with dropout
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer to map hidden state to the output dimension
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # LeakyReLU: allows small negative gradients to flow, alleviating problems caused by sparsity
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        
        # Store sequence length for generating sequence in decoder
        self.sequence_length = sequence_length
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)  # Initialize biases to 0
                
    def forward(self, x):
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        
        # Initialize decoder input as zeros (batch_size, 1, input_dim)
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        
        # Initialize output tensor list
        outputs = []
        
        # Decode step-by-step
        for _ in range(self.sequence_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.relu(self.fc(decoder_output))   # Map to input dimension
            outputs.append(output)
            decoder_input = output  # Set the next input to be the current output
        
        # Concatenate outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, sequence_length, input_dim)
        return outputs

    
class WeightedLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(WeightedLSTMAutoencoder, self).__init__()
        
        # Feature weighting layer
        self.feature_weighting = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.constant_(self.feature_weighting.weight[:, 2], 10)  # Initialize weight for the third column higher

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        self.sequence_length = sequence_length

    def forward(self, x):
        # Apply feature weighting
        x = self.feature_weighting(x)
        
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        
        # Initialize decoder input
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        outputs = []
        
        # Decode step-by-step
        for _ in range(self.sequence_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            outputs.append(output)
            decoder_input = output
        
        # Concatenate outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs
