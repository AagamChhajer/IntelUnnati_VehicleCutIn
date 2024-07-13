import torch.nn as nn

class LaneChangeDetectionModel(nn.Module):
    def __init__(self, feature_size, rnn_hidden_size, rnn_layers, output_classes):
        super(LaneChangeDetectionModel, self).__init__()
        self.rnn = nn.LSTM(feature_size, rnn_hidden_size, rnn_layers, batch_first=True)
        self.dense = nn.Linear(rnn_hidden_size, output_classes)

    def forward(self, input_seq):
        h0 = torch.zeros(rnn_layers, input_seq.size(0), rnn_hidden_size).to(input_seq.device)
        c0 = torch.zeros(rnn_layers, input_seq.size(0), rnn_hidden_size).to(input_seq.device)
        rnn_out, _ = self.rnn(input_seq, (h0, c0))
        output = self.dense(rnn_out[:, -1, :])
        return output

# Parameters
feature_size = 128  # Adjust based on the feature vector size
rnn_hidden_size = 64
rnn_layers = 2
output_classes = 2  # Binary classification: Cut-in or not

# Initialize the model
lane_change_model = LaneChangeDetectionModel(feature_size, rnn_hidden_size, rnn_layers, output_classes)
