import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

with open('input.txt', 'r') as file:
	text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

#given a string, converts each char into token id
def encode(input_string):
	return [stoi[char] for char in input_string]

#given list of token ids, converts each id into char 
def decode(token_ids):
	return ''.join([itos[id] for id in token_ids])

#print(encode("! \n this is encoding"))
#print(decode(encode("! \n this is encoding")))

#text -> token ids -> tensors 
data = torch.tensor(encode(text), dtype=torch.long)

#Split into train and test 
n = int(0.9*len(data))
train = data[:n]
test = data[n:]

'''
randomly selected chunks that goes into transformers
after block size truncate, transformer never receives more
'''
block_size = 8 
'''
num independent sequences processed in parallel 
occurs for every forward, backward pass
'''
batch_size = 64

def get_batch(split):
	batch_data = train if "train" in split else test
	#randomly generate numbers (size = batch_size)
	low, high = 0, len(batch_data) - block_size
	rand_vals = torch.randint(low, high, (batch_size,))
	#print(low)
	#print(high)
	#print(rand_vals)

	x_temp, y_temp = [], [] #list of tensors, must change memory-inefficient 

	#Get block_size starting at rand int from data for x 
	#y is one offset of x (predictions)
	for val in rand_vals:
		x_temp.append(batch_data[val:val+block_size])
		y_temp.append(batch_data[val + 1:val+block_size + 1])

	#stack appends in new dimension, cat appends in given dimension
	x = torch.stack([t for t in x_temp])
	y = torch.stack([t for t in y_temp])
	#print(x)
	#print(y)
	return x, y
	
xb, yb = get_batch("train")
print(xb.shape)
print(yb.shape)
n_embed = 32
#head_size = 16
class Head(nn.Module):

	def __init__(self, head_size):
		super().__init__()
		#key, query, value 
		self.key = nn.Linear(n_embed, head_size, bias=False)
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		tril = torch.tril(torch.ones(block_size, block_size))
		self.register_buffer("tril", tril)

	def forward(self, x):
		B, T, C = x.shape
		x_key = self.key(x)
		x_query = self.query(x)

		x_weight = x_query @ x_key.transpose(-2, -1)
		#normalize the weights "scaled attention"
		x_weight = x_weight * C**-0.5 #TODO
		#decoder block 
		x_weight = x_weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		x_weight = F.softmax(x_weight, dim=-1)
		#value aggregation
		x_value = self.value(x)
		out = x_weight @ x_value

		return out 

#basically multiple heads running in parallel (indepedent) to faciliatate communication
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList(Head(head_size) for head in range(num_heads))
		self.proj = nn.Linear(n_embed, n_embed) #residual connection

	def forward(self, x):
		out = torch.cat([head(x) for head in self.heads], dim=-1)
		out = self.proj(out) #linear transform of layer (resid conn)
		return out

#per token level 
class FeedForward(nn.Module):
	def __init__(self, n_embed):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed, 4 * n_embed), 
			nn.ReLU(),
			nn.Linear(4 * n_embed, n_embed)
		)
	
	def forward(self, x):
		return self.net(x)
	

class Block(nn.Module):
	def __init__(self, n_embed, n_heads):
		super().__init__()
		head_size = n_embed // n_heads
		self.self_attn = MultiHeadAttention(n_heads, head_size)
		self.feed_forward = FeedForward(n_embed)
		self.layer_norm_1 = nn.LayerNorm(n_embed)
		self.layer_norm_2 = nn.LayerNorm(n_embed)
	
	def forward(self, x):
		#residual connection (adding on)
		x = x + self.self_attn(self.layer_norm_1(x))
		x = x + self.feed_forward(self.layer_norm_2(x))
		return x

class BigramLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()
		#create a "token embedding table" -> lookup table (dimensins = vocab_size x vocab_size)
		self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
		#positional embedding 
		self.position_embedding_table = nn.Embedding(block_size, n_embed)
		self.blocks = nn.Sequential(
			Block(n_embed, n_heads=4), 
			Block(n_embed, n_heads=4), 
			Block(n_embed, n_heads=4),
			nn.LayerNorm(n_embed),
		)
		self.feed_forward = FeedForward(n_embed)
		#Embeddings to logits requires a linear layer 
		self.lang_model_head = nn.Linear(n_embed, vocab_size)
		#self.self_attn_head = Head(n_embed)
		self.self_attn_head = MultiHeadAttention(4, n_embed//4) #4 heads of 8 dimensions

	
	#forward -> take inputs of x into the lookup table
	def forward(self, index, targets=None):
		#dimensions = (batch X time X channel)
		#OR (batch_size X block_size X vocab_size)
		B, T = index.shape

		token_embeddings = self.token_embedding_table(index) #(B, T, C)
		position_embeddings = self.position_embedding_table(torch.arange(T))
		x = token_embeddings + position_embeddings
		x = self.self_attn_head(x)
		x = self.feed_forward(x)
		logits = self.lang_model_head(x) #(B, T, vocab_size)
		
		if targets is None:
			loss = None
		else:
			#[theoretical] loss = -ln(1/vocab_size)
			B, T, C = logits.shape
			#print(logits.shape)
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)
		return logits, loss 

	#generate 
	def generate(self, index, max_new_tokens):
		#index dim = batch X time, represents current context 
		for token in range(max_new_tokens):
			#condition ensure index is <= block size 
			index_resized = index[:, -block_size:]
			#generate predictions
			logits, loss = self.forward(index_resized)
			logits = logits[:, -1, :] #remove one from the "time" dimension (B, C)
			probs = F.softmax(logits, dim=-1) # B, C
			next_index = torch.multinomial(probs, num_samples=1) # B, 1
			index = torch.cat((index, next_index), dim=1)
		return index

model = BigramLanguageModel()
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #optimizes step size 
print("parameters ", sum(p.numel() for p in model.parameters())) 
max_iters = 100000
eval_interval = 3000
eval_iters = 2000

@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ["train", "test"]:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			x, y = get_batch(split)
			logits, loss = model(x, y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

for iter in range(max_iters):
	if iter % eval_interval == 0:
		losses = estimate_loss()
		print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
	
	xb, yb = get_batch("train")
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()

index = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(index, max_new_tokens=1000)[0].tolist()))
