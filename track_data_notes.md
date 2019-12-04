    class Model(nn.Module):
        def __init__(self, input_size, output_size, hidden_dim, n_layers):
            super(Model, self).__init__()
    
            # Defining some parameters
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
    
            # Defining the layers
            # RNN Layer
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)  #batch_first – If True, then the input and output tensors are provided as
                                                                                    # (batch, seq, feature). Default: False
            # Fully connected layer
            self.fc = nn.Linear(hidden_dim, output_size)
    
        def forward(self, x):
            batch_size = x.size(0)
    
            # Initializing hidden state for first input using method defined below
            hidden = self.init_hidden(batch_size)
    
            # Passing in the input and hidden state into the model and obtaining outputs
          (1)  out, hidden = self.rnn(x, hidden)
    
            # Reshaping the outputs such that it can be fit into the fully connected layer
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
    
            return out, hidden
    
        def init_hidden(self, batch_size):
            # This method generates the first hidden state of zeros which we'll use in the forward pass
            # We  78 8i 'll send the tensor holding the hidden state to the device we specified earlier as well
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
            return hidden
        
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        input_seqs=input_seqs.cuda()
        output, hidden = model(input_seqs)
        # target_res = target_seqs.view(-1).long()
        loss = criterion(output, target_seqs)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

# ERRO
(1)<class 'tuple'>: (<class 'RuntimeError'>, RuntimeError('cuDNN error: CUDNN_STATUS_BAD_PARAM',), 
<traceback object at 0x000001278F3D7E48>)

* solution:
I had the same error. it was caused by the date type of the input data. 
The model expected float32, while my input was float64.
So,you should change the type of 'input_seqs' and 'target_seqs' to torch.float32

        input_seqs = torch.from_numpy(input_seqs)
        target_seqs = torch.Tensor(target_seqs)
        input_seqs,target_seqs = input_seqs.type(torch.float32),target_seqs.type(torch.float32)

(2)<class 'tuple'>: (<class 'RuntimeError'>, RuntimeError('The size of tensor 
a (2) must match the size of tensor b (24880) at non-singleton dimension 1',), <traceback object at 0x000001D382F36A88>)

    target_res = target_seqs.view(-1).long()
    loss = criterion(output, target_res)
    
* 

(3)<class 'tuple'>: (<class 'RuntimeError'>, RuntimeError("Expected object of backend CUDA but got backend CPU for argument #2 'target'",), <traceback object at 0x000001EDF26A4048>)
    
    target_res = target_seqs.view(-1).long()
    loss = criterion(output, target_seqs)

*breakpoint*:

(1)

hidden： torch.Size([1, 1244, 12])

x：torch.Size([1244, 10, 40])

(2)
hidden: torch.Size([1, 500, 32])
input:  torch.Size([500, 10, 40])


(2)     
    parameters shape b
    input_seqs      torch.Size([1244, 10, 40])
    
>>>
output:
tensor([[1.0969, 0.2227],
        [1.4701, 0.2375],
        [0.7403, 0.5075],
        ...,
        [1.7095, 0.2786],
        [0.7148, 0.5136],
        [1.1526, 0.4966]], device='cuda:0', grad_fn=<AddmmBackward>)
        
torch.Size([12440, 2])

target_seqs:  tensor([[10.4460, 10.7370],
        [13.2530,  3.2630],
        [10.4740,  1.1810],
        ...,
        [14.2270,  8.2730],
        [21.1240,  1.1810],
        [26.3420,  1.1810]])
torch.Size([12440, 2])

output:
tensor([[-0.2712, -0.3859],
        [-0.3432, -0.4444],
        [-0.1327, -0.3132],
        ...,
        [-0.5711, -0.2989],
        [-0.2742,  0.0603],
        [-0.3457,  0.0870]], device='cuda:0')
torch.Size([12440, 2])