import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize(bits,gradient_bits,g_q=False): # g_q is True if gradient will be quantized
    class Quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            if bits==1:
                # torch.sign returns tenary values [-1,0,1], so output 0 should be map to 1 or -1
                input=torch.sign(input)
                input[input==0]=1 
            elif bits==32:
                return input
            else:
                # Equation 5
                level=float(2**bits-1)
                input=torch.round(input*level)/level
            return input

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            if not g_q and bits==1:
                ## No Gradient Quantization, and if bits==1
                ## STE backward pass
                mask=torch.abs(input)<=1
                grad_input=grad_input*mask
            if not g_q:
                return grad_input
            else:
                # Quantization Gradient
                # Equation 12
                rank = len(grad_input.shape)
                assert rank is not None
                maxx = torch.max(torch.abs(grad_input), 1, keepdim=True) # max(|dr|)
                for i in range(2, rank):
                    maxx = torch.max(torch.abs(maxx[0]), i, keepdim=True)
                grad_input=grad_input/maxx[0] # dr/max(|dr|)
                level=float(2**gradient_bits-1)
                grad_input=grad_input*0.5+0.5+ torch.FloatTensor(grad_input.shape).uniform_(-0.5 / level, 0.5 /level).cuda()
                grad_input=torch.clamp(grad_input,0.0,1.0) # Clamp for quantization gradient input
                grad_input=torch.round(grad_input*level)/level-0.5  # Quantization -1/2
                grad_input=grad_input*2*maxx[0]
            return grad_input
    return Quantize().apply

class Quantizer(nn.Module):
    ## In/Output should be range [0,1]
    def __init__(self,bits,gradient_bits,g_q):
        super(Quantizer,self).__init__()
        self.bits =  bits
        self.gradient_bits=gradient_bits
        self.g_q=g_q # True if Quantize Gradient
        self.quantize=quantize(self.bits,self.gradient_bits,self.g_q)
    def forward(self,x):
        # if self.bits!=1:
        #     assert torch.max(x)<=1.0 and torch.min(x)>=0.0, 'Input should be range between 0 and 1'

        return self.quantize(x)
  
        
class Quantization_weight(nn.Module):
    def __init__(self,w_bits,g_bits,g_q):
        super(Quantization_weight, self).__init__()
        self.w_bits=w_bits
        self.g_bits=g_bits
        self.quantizer=Quantizer(self.w_bits,g_bits,g_q=g_q)
    def forward(self,x):
        if self.w_bits==1:
            mean=torch.mean(torch.abs(x)).detach()
            x=self.quantizer(x/mean)*mean
        elif self.w_bits==32:
            return x
        else:

            # X should be range [0,1]
            tanh=torch.tanh(x)
            maxx=torch.max(torch.abs(x)).detach()
            x=tanh/(2*maxx)+0.5
            x=2*self.quantizer(x)-1
            assert torch.abs(x)<=1, 'output weight should be range [-1,1] '
        return x
class QuantizationActivation(nn.Module):
    def __init__(self,a_bits,g_bits,g_q):
        super(QuantizationActivation, self).__init__()
        self.a_bits=a_bits
        self.g_bits=g_bits
        self.g_q=g_q
        self.quantizer=Quantizer(self.a_bits,self.g_bits,self.g_q)
    def forward(self,x):
        if self.a_bits==32:
            return x
        else:
            # Quantizer Input should be range [0,1]
            x=self.quantizer(torch.clamp(x,0.0,1.0))
            return x

class QuantizationFullyConnected(nn.Linear):
    def __init__(self,in_channel,out_channel,bias=True,w_bits=1,g_bits=32,g_q=False):
        super(QuantizationFullyConnected, self).__init__(in_channel,out_channel,bias)
        self.w_bits=w_bits
        self.g_bits=g_bits
        self.g_q=g_q
        self.quantization_weight=Quantization_weight(w_bits,g_bits,g_q)
    def forward(self,x):
        # Weight 양자화
        quantized_weight=self.quantization_weight(self.weight)
        # Fully Connected 연산 with quantized weight
        return F.linear(x,quantized_weight,self.bias)
    

class QuantizationConv2d(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,w_bits=1,g_bits=32,g_q=False):
        super(QuantizationConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                    padding, dilation, groups, bias)
        self.quantization_weight=Quantization_weight(w_bits,g_bits,g_q)
    def forward(self,x):
        # Weight 양자화
        quantized_weight=self.quantization_weight(self.weight)
        # Convolution 연산 with quantized weight
        return F.conv2d(x,quantized_weight,self.bias,self.stride,self.padding,self.dilation,self.groups)