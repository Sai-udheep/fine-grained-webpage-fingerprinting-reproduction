import torch
import torch.nn as nn
import torch.nn.functional as F


class DF(nn.Module):
    def __init__(self, emb_size, final_length):
        super(DF, self).__init__()
        filter_num = ['None', 32, 64, 128, 256]
        kernel_size = ['None', 8, 8, 8, 8]
        conv_stride_size = ['None', 1, 1, 1, 1]
        pool_stride_size = ['None', 4, 4, 4, 4]
        pool_size = ['None', 8, 8, 8, 8]
        pool_padding = 0

        self.feature_extraction = nn.Sequential(
            # block1
            nn.Conv1d(in_channels=1,out_channels=filter_num[1],kernel_size=kernel_size[1],
                      stride=conv_stride_size[1],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv1d(in_channels=filter_num[1],out_channels=filter_num[1],kernel_size=kernel_size[1],
                      stride=conv_stride_size[1],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_padding),
            nn.Dropout(p=0.1),


            # block2
            nn.Conv1d(in_channels=filter_num[1],out_channels=filter_num[2],kernel_size=kernel_size[2],
                      stride=conv_stride_size[2],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=filter_num[2],out_channels=filter_num[2],kernel_size=kernel_size[2],
                      stride=conv_stride_size[2],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_padding),
            nn.Dropout(p=0.1),

            # block3
            nn.Conv1d(in_channels=filter_num[2],out_channels=filter_num[3],kernel_size=kernel_size[3],
                      stride=conv_stride_size[3],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=filter_num[3],out_channels=filter_num[3],kernel_size=kernel_size[3],
                      stride=conv_stride_size[3],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_padding),
            nn.Dropout(p=0.1),
            
            # block4
            nn.Conv1d(in_channels=filter_num[3],out_channels=filter_num[4],kernel_size=kernel_size[4],
                      stride=conv_stride_size[4],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=filter_num[4],out_channels=filter_num[4],kernel_size=kernel_size[4],
                      stride=conv_stride_size[4],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_padding),
            nn.Dropout(p=0.1),
        )

        ### Replace the original  fully connected layers with a linear layer 
        self.generate_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=filter_num[4]*final_length, out_features=emb_size)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.generate_feature(x)
        return x
