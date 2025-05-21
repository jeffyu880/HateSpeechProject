import torch 
import torch.nn as nn
import torch.nn.functional as F

class MM(nn.Module):
    def __init__(self, n_out):
        super(MM, self).__init__()  
        
        self.dense_vgg_1024 = nn.Linear(4096, 1024)
        self.dense_vgg_512 = nn.Linear(1024, 512)
        self.drop20 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.05) 
        
        self.dense_drob_512 = nn.Linear(768, 512)
        
        self.gen_key_L1 = nn.Linear(512, 256) # 512X256
        self.gen_query_L1 = nn.Linear(512, 256) # 512X256
        self.gen_key_L2 = nn.Linear(512, 256) # 512X256
        self.gen_query_L2 = nn.Linear(512, 256) # 512X256
        self.gen_key_L3 = nn.Linear(512, 256) # 512X256
        self.gen_query_L3 = nn.Linear(512, 256) # 512X256
#         self.gen_value = nn.Linear(512, 256) # 512X256
        self.soft = nn.Softmax(dim=1)
        self.soft_final = nn.Softmax(dim=1)
        self.project_dense_512a = nn.Linear(1024, 512) # 512X256
        self.project_dense_512b = nn.Linear(1024, 512) # 512X256
        self.project_dense_512c = nn.Linear(1024, 512) # 512X256 
        
        
        self.fc_out = nn.Linear(512, 256) # 512X256
        self.out = nn.Linear(256, n_out) # 512X256
        

    def selfattNFuse_L1a(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L1(vec1))
            k1 = F.relu(self.gen_key_L1(vec1))
            q2 = F.relu(self.gen_query_L1(vec2))
            k2 = F.relu(self.gen_key_L1(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.project_dense_512a(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    def selfattNFuse_L1b(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L2(vec1))
            k1 = F.relu(self.gen_key_L2(vec1))
            q2 = F.relu(self.gen_query_L2(vec2))
            k2 = F.relu(self.gen_key_L2(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.project_dense_512b(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    
    def selfattNFuse_L2(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L3(vec1))
            k1 = F.relu(self.gen_key_L3(vec1))
            q2 = F.relu(self.gen_query_L3(vec2))
            k2 = F.relu(self.gen_key_L3(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.project_dense_512c(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep


    def forward(self, in_CI, in_VGG, in_CT, in_Drob):        
        VGG_feat = self.drop20(F.relu(self.dense_vgg_512(self.drop20(F.relu(self.dense_vgg_1024(in_VGG))))))
        Drob_feat = self.drop5(F.relu(self.dense_drob_512(in_Drob))) # distilled text?
        out_img = self.selfattNFuse_L1a(VGG_feat, in_CI)
        out_txt = self.selfattNFuse_L1b(Drob_feat, in_CT)        
        out_img_txt = self.selfattNFuse_L2(out_img, out_txt)
        final_out = F.relu(self.fc_out(out_img_txt))
        out = torch.sigmoid(self.out(final_out)) #For binary case
        # out = self.out(final_out)
        return out

    