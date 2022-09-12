import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleNet(nn.Module):
    def __init__(self, varnet, multidomainnet, crossdomainnet7, crossdomainnet8):
        super(EnsembleNet, self).__init__()

        self.varnet = varnet
        self.multidomainnet = multidomainnet
        self.crossdomainnet7 = crossdomainnet7
        self.crossdomainnet8 = crossdomainnet8
        
#        self.v_param = nn.Parameter(torch.ones(1)/4)
#        self.m_param = nn.Parameter(torch.ones(1)/4)
#        self.c7_param = nn.Parameter(torch.ones(1)/4)
        self.v_param = 0.08
        self.m_param = 0.65
        self.c7_param = 0


    def forward(self, kspace, mask):
        varnet_output = self.varnet(kspace, mask)
        multidomainnet_output = self.multidomainnet(kspace, mask)
        crossdomainnet7_output = self.crossdomainnet7(kspace, mask)
        crossdomainnet8_output = self.crossdomainnet8(kspace, mask)
       
        output = (varnet_output+multidomainnet_output+crossdomainnet7_output+crossdomainnet8_output)/4
        #c8_param = 1-self.v_param-self.m_param-self.c7_param
        #output = varnet_output*self.v_param + multidomainnet_output*self.m_param + crossdomainnet7_output*self.c7_param + crossdomainnet8_output*c8_param
        return output
