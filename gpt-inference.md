# GPT Full Inference  
1. A sequence of text is the input. 
2. Each input is then tokenized and unique IDs are assigned to each token by the tokenizer. 
3. Each token needs to be embedded into a high dimensional space 
4. First dropout occurs to prevent over-fitting 
5. Input is then passed into 12 layers of transformer block. 
6. Each transformer block goes through the sequence of Layer Normalization 1, Masked Multi-Head Attention, Dropout, Residual Add. Layer Normalization 2, Feed Forward, Dropout, Residual Add. 
7. Note that the main difference between these two pass inside a transformer block is Masked Multi-Head Attention is replaced by a Feed Forward Layer in the second pass. 
8. Each normalization goes through the same process. LayerNorm normalizes across the features dimensions across one vector. Find the Mean, Variance, Standard Deviation. LayerNorm prevents the numbers inside a token vector from becoming too large, too small, or badly shifted.

$ "LayerNorm: "(x_i) = gamma_i dot (x_i - mu) / sqrt(sigma^2 + epsilon) + beta_i $
$ "Mean: " mu = 1/d sum_(j=1)^d x_j $
$ "Variance: " sigma^2 = 1/d sum_(j=1)^d (x_j - mu)^2 $
$ "Input: " x = [x_1, x_2, ..., x_d] $


9. Multi-Head attention takes one normalized input matrix and feeds it into multiple attention heads. Keep In mind that within the GPT architecture, there are multiple layers of multi-head attention modules. 

10. Each attention head consists of Q, K, V projection _weight matrix_. Each row corresponds to one input embedding dimension. It contains learned weights that say how that input dimension contributes to each output dimension after creating the new vector. The number of columns represents the feature space where attention comparison begins. The output is a projected versions of the input embeddings, created using learned weight matrices. 

11. Query, Key, and Value are matrices. For each row, it represents a projected version of a token embedding. Between the Query and Key matrix a matrix multiplication operation has to happen. This process finds the similarity score between a query vector against key vector. The final output of this would be a attention score matrix.  

12. Assume that the word "it" is the query, and that contextually it is the most similar to the word "animal". There is an association between "it" and "animal", however note that "animal" key vector is not the value. The word "animal" is the token whose key vector matched the query vector strongly. So we should provide the corresponding value vector that represents the word "animal". 

13. Getting the attention score matrix, we apply the Softmax function for each cell within the matrix. This process normalizes the output such that the set of numbers sum up to 1, thus creating the attention weight matrix from the attention score matrix. 

13. After the attention score matrix has been computed, we don't actually provide anything from the vector matrix as stated in step 12. This is just an intuitive language. Mechanically, we multiply the attention weight matrix with the Value Matrix to get the context vector matrix. This process mixes the actual value information from the attention weights into the value matrix to get the context vectors. 

14. Each row in the value matrix represents one token as a projected value vector. Together, all rows in the value matrix represent the sequence in value-space. Each row in the attention-weight matrix corresponds to one query token and describes how strongly that token attends to every token in the sequence. When we take one row from the attention-weight matrix and multiply it by the full value matrix, we compute a weighted sum of all value vectors. The result is the context vector for that query token.

15. The repeated process of creating the context vector results in a final context matrix. This process between steps 9 - 15 will happen for each independent attention head. Each head will result in a unique context matrix. At the end of multi-head attention, we concatenate the context matrices from all heads into one combined context matrix. 

16. In the last step, we take the concatenated/flattened context matrix and multiply it by Output Weight Matrix to get the final multi-head attention output. The Output Weight Matrix is just another learned weight matrix that lets the model mix information across heads.

17. Apply the Dropout technique and apply the residual connection where we add the original input matrix to get the final output before passing it into another Layer Normalization for the second pass. 

18. After the second layer normalization we feed the output matrix into the feed-forward network. The feed-forward network transforms each token vector independently, usually by expanding the feature dimension, applying a nonlinear activation such as GELU, and then contracting by projecting it back to the original embedding dimension.

19. The first expansion happens in the first linear layer. Suppose each token vector has 4 dimensions. Now suppose the first feed-forward linear layer has the shape (4, 12). Through this matrix multiplication (1, 4) @ (4, 12), we have expanded the vector representation into a higher dimension of (1, 12). This expansion happens for each input vector. 

20. For each row, each dimension gets passed into a GELU function. The GELU function is a non-linear function that bends each coefficient to describe how strongly that feature should pass through. 

21. After applying the GELU function, we then contract the matrix back into the original dimensions where the input dimension of the second linear layer is greater than its output dimension. Apply the dropout technique and then apply the residual connection, then you have the output of the transformer block. 

22. The output of the transformer block is also called the hidden states or updated token representations. Take the hidden state matrix and pass it through a final linear layer that maps each token to the wight vocab matrix. So each row becomes a score over the entire vocabulary. hidden_state.shape = (num_tokens, 768), W_vocab.shape = (768, vocab_size). Resulting in (num_token, vocab_size). This new matrix is called logits. 

23. The model produces logits for every position in the sequence, but when generating the next token, we usually use only the logits from the last position. These last-position logits are passed through a softmax function, which converts the raw scores into a probability distribution over the vocabulary. After softmax, all probabilities are between 0 and 1 and sum to 1, allowing the model to choose or sample the next token. 

24. We take the highest-probability position from the probability vector. The position/index of that score corresponds to a token ID. We then take that token ID and use the tokenizer’s vocabulary/dictionary to convert the ID back into text.