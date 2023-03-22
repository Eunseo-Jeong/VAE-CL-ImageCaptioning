import torch 
import torch.nn as nn
from torch import einsum
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, GPT2LMHeadModel, ViTModel, GPT2Config
from transformers import ViTConfig, GPT2Config
# from transformers import BertModel, ViTFeatureExtractor, ViTModel, BartModel, AutoTokenizer, BartForConditionalGeneration, GPT2TokenizerFast
# from transformers import T5Model, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTConfig, BertConfig, RobertaModel, EncoderDecoderModel
import torch.nn.functional as F
from config import config_data

from visionEncoderDecoderConfig import VisionEncoderDecoderConfig
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone() # 마지막 pad token 제거

    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id
    
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id) # mask가 0인 부분을 pad_token_id로 채우기

    return shifted_input_ids




class ModelClass(nn.Module):

    def __init__(self, config):
        super(ModelClass, self).__init__()

        vit_config = ViTConfig()
        gpt2_config = GPT2Config()
        
        self.vision_config = vit_config
        self.decoder_config = gpt2_config
        self.decoder_config.add_cross_attention = True

        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_decoder = GPT2LMHeadModel.from_pretrained('gpt2', config=self.decoder_config)

        # self.visionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning") # "google/vit-base-patch16-224-in21k", "gpt2"
        # self.vision_encoder = self.visionEncoderDecoderModel.encoder # vit
        # self.text_decoder = self.visionEncoderDecoderModel.decoder # gpt2 LMhead (layer_norm -> linear layer(768, 50257)) 
    
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        # self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        additional_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>'}) # 'bos_token' : '<bos>'
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
        
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) # 50257 -> 50258

        # self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>") #  
        # self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<bos>") # 
        # self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<eos>") # 
        # self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<unk>") # 
        
        # print("pad_token_id: ", self.tokenizer.pad_token_id, "bos_token_id", self.tokenizer.bos_token_id, "eos_token_id", self.tokenizer.eos_token_id)
        self.config = config
        self.latent_size = self.config['latent_size']
        
        self.hidden_size = self.vision_config.hidden_size # 768
        self.decoder_hidden_size = self.decoder_config.n_embd # 768

        self.vocab_size = len(self.tokenizer)

        self.relu = nn.ReLU()


        # self.lm_head = nn.Linear(self.hidden_size, self.vocab_size) # 768, vocab_size

        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_size) # 25000, 768
        # self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)

        self.hidden2mean = nn.Linear(self.hidden_size, self.latent_size) # 768, 64
        self.hidden2logv = nn.Linear(self.hidden_size, self.latent_size) # 768, 64

        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size) # 64, 768
        # self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab) # 768, 2523
    
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.contrastive_loss_weight = 1.

        self.enc_to_dec_proj = nn.Linear(self.hidden_size, self.hidden_size) # encoder hidden_size, decoder hidden_size

        self.eosLinear = nn.Linear(len(self.tokenizer), 768)


    def reconstruction_loss(self, x, label): # batch, seq, 50265 / batch, seq
        # loss = nn.CrossEntropyLoss()
        output = self.loss_fct( # F.cross_entropy
            x.transpose(1, 2), # batch, 50265, seq 
            label, # batch, seq
            # ignore_index=self.tokenizer.pad_token_id,
            # reduction="none",
        )
        return output
    
    # 이상적인 sampling 함수 값이 최대한 prior과 같도록
    def regularization_loss(self, mu, logvar, training=False): 
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()) # batch, 64
        # if self.min_z and training:
            # dimensionwise_loss[dimensionwise_loss < self.min_z] = self.min_z
        loss = KLD.sum(-1) # batch
        return loss

    def reparameterize(self, mu, logvar): # batch, 64
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 정규분포 상의 랜덤 값으로 초기화 (사이즈는 std와 동일)
        return mu + std * eps
        
    def calculate_latent(self, pooled): # last_hidden_state (batch, 197, 768)
        # print(pooled.shape)
        mu, logvar = self.hidden2mean(pooled), self.hidden2logv(pooled) # (batch, 197, 768) (768, 64) -> batch, 197, 64 
        # print(mu.shape, logvar.shape)

        mu, logvar = self.relu(mu), self.relu(logvar)
        z = self.reparameterize(mu, logvar) # batch, 197, 64  (z 생성과정에 zero-mean 가우시안으로 뽑은 노이즈 개입)
        # print(z.shape)
        return z, mu, logvar

    def forward(self, mode, device, images, captions): 
        device = device

        caps_tok = self.tokenizer(captions, padding='longest', return_tensors='pt')
        caps_input_ids = caps_tok['input_ids'].to(device)
        caps_attention_mask = caps_tok['attention_mask'].to(device)
        
        caption_len = caps_input_ids.size()[1]
        batch_size = caps_input_ids.size()[0]
        
        caps_eos = caps_input_ids[:, 1:]

        # print( self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.unk_token_id )


        ############################### image ###############################
        pixel_values = self.feature_extractor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device) # torch.Size([64, 3, 224, 224])
        
        # print(pixel_values.shape)

        vit_encoder = self.vision_encoder(
            pixel_values = pixel_values
        ) # last_hidden_state, pooler_output
        encoder_hidden_states = vit_encoder['last_hidden_state'] # batch, 197, 768
        vit_cls = vit_encoder.last_hidden_state[:,0:1,:] # CLS (batch, 1, 768) 

        # print(encoder_hidden_states.shape)
        # print(vit_cls.shape)

        
        if self.config['cls_latent_vector'] == True:
            image_z, image_mu, image_logvar = self.calculate_latent(vit_cls) # make z latent vector from image last_hidden_states
        else:
            image_z, image_mu, image_logvar = self.calculate_latent(encoder_hidden_states)
            
        image_reg_loss = self.regularization_loss(image_mu, image_logvar) # image regularization loss

        # print(encoder_hidden_states.shape) # 64, 197, 768
        # print(image_z.shape) # 64, 197, 64

        ############################################################################################################
        # print(image_z.size()[-1], self.decoder_hidden_size)
        if image_z.size()[-1] != self.decoder_hidden_size:
            image_z = self.latent2hidden(image_z) # (batch, 197, 64) (64, 768) -> (batch, 197, 768) # text 생성에 입력으로 들어가야 함
        # print(image_z.shape)
        image2hidden = self.enc_to_dec_proj(image_z) # batch, 197, 768 -> batch, 197, 768
        # print(image2hidden.shape)
        

        labels = caps_input_ids # bos 사용하지말고 eos 써야함
        # decoder_input_ids = shift_tokens_right(text_input_ids, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id) # question + answer
        decoder_input_ids = caps_input_ids
        decoder_attention_mask = caps_attention_mask

        # print(decoder_input_ids)
        # print(decoder_attention_mask)

        # print(decoder_input_ids.shape)
        # print(decoder_attention_mask.shape)

        return_dict = True

        decoder_outputs = self.text_decoder( # GPT-2 # input_ids, past_key_values, attention_mask, token_type_ids , position_ids, head_mask, input_embeds, use_cache
            input_ids=decoder_input_ids, # question + text
            attention_mask=decoder_attention_mask, # attention
            encoder_hidden_states=image2hidden, # (batch, seq_len, hidden_size) -> vit last_hidden_state
            # encoder_attention_mask=encoder_attention_mask,
            # inputs_embeds=decoder_inputs_embeds, # (batch, target_seq_len, hidden_size)
            # output_attentions=output_attentions,
            output_hidden_states=True,
            # use_cache=use_cache,
            # past_key_values=past_key_values,
            return_dict=return_dict # true = transformers.file_utils.ModelOutput
        ) # logits
        
        # print(decoder_outputs.logits.shape)  # torch.Size([64, 26, 50258])
        text_decoder_last_hidden_state = decoder_outputs.hidden_states[-1] # torch.Size([64, 26, 768])
        eos_embedding = text_decoder_last_hidden_state[:, 1:, :] # torch.Size([64, 4, 768])


        caps_eos_idx = torch.where(caps_eos==self.eos_token_id, 1., 0.)
        # print(caps_eos_idx.shape) # torch.Size([64, 4])
        # print(caps_eos_idx) 

        caps_eos_idx = caps_eos_idx.unsqueeze(dim=1) # torch.Size([64, 1, 4]) # label
        # print(caps_eos_idx.shape)
        # print(eos_embedding.shape)
        # print()
        eos_out = torch.bmm(caps_eos_idx, eos_embedding) # batch matrix 
        # print(eos_out.shape)



        loss = None
        if labels is not None: # logits: 64, seq, 50259 -> 64, 7, 50259 / label: 64, 7
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            # logits = decoder_outputs.logits[:, caption_len:, :] # 
            
            # print(logits.shape) # remove the back eos token 
            # print(logits[:,:-1,:].shape)

            
            # print(labels.shape) # remove the front eos token
            # print(labels[:,1:].shape)
            
            loss = self.loss_fct(logits[:,:-1,:].reshape(-1, self.vocab_size), labels[:,1:].reshape(-1)) # 448, 50259, # 448
            # print(loss)
            # quit()

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + vit_encoder
            else:
                return decoder_outputs + vit_encoder
        
        out = Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states, # None
            decoder_attentions=decoder_outputs.attentions, # None
            cross_attentions=decoder_outputs.cross_attentions, # None
            encoder_last_hidden_state= vit_encoder.last_hidden_state, # 64, 197, 768 vit_encoder.last_hidden_state, image2hidden
            encoder_hidden_states=vit_encoder.hidden_states, # None 
            encoder_attentions=vit_encoder.attentions # None 
        ) # loss, logits, past_key_values=None, decoder_hidden_states=None, decoder_attentions=None,
        # cross_attentions=None, encoder_last_hidden_state,encoder_hidden_states=None, encoder_attentions=None
        
       ############################################################################################################

        decoder_loss = out.loss # loss, logits, past_key_values, encoder_last_hidden_state
        logits = out.logits # batch, seq, 50258-> Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
        
        # recon_loss = self.reconstruction_loss(logits[:, 18:, :], labels)
        

        if mode == 'eval':
            return logits, caption_len, caps_input_ids, self.tokenizer
        
        

        ############################################################################################################
        # contrastive learning (vit_cls, decoder eos token)
        if self.config['contrastive'] == True:

            eos_out = eos_out.squeeze(1)
            vit_cls = vit_cls.squeeze(1)

            sim = einsum('i d, j d -> i j', eos_out, vit_cls) # batch, 768 / batch, 768 -> batch, batch
            sim = sim * self.temperature.exp()
            contrastive_labels = torch.arange(batch_size, device=device) # batch만큼 labels

            contrastive_loss = (self.loss_fct(sim, contrastive_labels) + self.loss_fct(sim.t(), contrastive_labels)) * 0.5  
            cls_contrastive_loss = contrastive_loss * self.contrastive_loss_weight
            # print(cls_contrastive_loss)
            # quit()
            
            return decoder_loss, cls_contrastive_loss, image_reg_loss.mean()
        
        else:
            return decoder_loss, image_reg_loss.mean()
        
        
        # logits = logits.reshape(-1, 50261) # 320, 50261
        # print(labels.view(-1).shape) # 320
