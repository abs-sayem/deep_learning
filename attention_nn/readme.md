### **Attention Neural Network**
**What is Attention?**
###### **Suppose I give you the `Deep Learning with Python` book and ask you to tell me about `neural style transfer` from the book. What would you do? - either start reading the whole book or check the index to find out where "neural style transfer" is talked about, go there and read the related portion. The second way is more precise because you draw the attention in the region of interest.**
**Understand Attention Mechanism**
###### **To understand attention mechanism, here we will consider `Image Captioning through Attention`. How we can caption an image?- The basic idea consists of 2 steps: First `Encode` the input image in an internal vector representation(H) using a `CNN` and then `Decode` the encoded representation(H) into word vectors that signifies the caption using a `RNN(say-LSTM)`.**
> ###### **`Image -- (Encoded by CNN) --> H -- (Decoded by LSTM) --> Caption`**

<img src="images/0.PNG" height="300" width="1000">

###### **`[NB]`The problem is, for captioning the image, LSTM consider the entire image vector representation(H) every time. This is not an efficient way, because- we generally caption a specific region not the whole image.**

<img src="images/1.PNG" height="300" width="1000">

###### **`How to solve this?`- We can create non-overlapping subregion of the image and focus on specific region. When decoder decides on a caption for every word it only looks at specific regions of the image. This leads to a more accurate description.**
> ###### **`Image -- (Encoded by CNN) --> (h_1.....h_n) -- (Decoded by LSTM) --> Word vector`**<br><img src="images/2.PNG" height="300" width="1000">

**But how does it exactly decide the region or regions to consider?**

**Attention Mechanism**
###### **An attention unit considers all the subregions(H) and contexts(C) as its input and outputs the weighted `arithmetic mean`(Z) of these regions.<br><img src="images/4.PNG" height="300" width="1000"><br>**`What is Arithmetic Mean?`- the inner product of actual values and their probabiliies.<br><img src="images/arithmatic_mean.PNG"><br>`How the Probabilities and Weights deternmine?`- using the `context`(C). `What is context?`- Context represents everything that `RNN` has output.**<br><img src="images/prob_weigh.PNG" height="300" width="1000">
**Attention Unit**
###### **We have- inputs(y) from CNN and context(C) from RNN. These inputs then applied to the weights which constitute the learn about parameters of the attention unit. This means the weight vectors update as we get more training data.`??`**<br><img src="images/7.PNG" height="300" width="1000">
> ###### **`m_1 = tanh(y_1.w_1 + C.w_c)`**
###### **the `tanh` activation fn scales the values between (-1 to 1), which leads to a much smoother choice of regions-of-interest within each sub-region.`???`**<br><img src="images/10.PNG" height="300" width="1000">
###### **`[NB]` We don't necessarily have to apply a tanh function, we only need to ensure the regions that we output are relevant to the context. We can choose regions-of-interest by applying a simple dot product of regions(y) and context(c).<br><img src="images/11.PNG" height="300" width="1000"><br>The higher the product, the more similar they are. The difference between using the simple dot product and tanh function would be granuality(level of details in a set of data). Tanh is more fine-grained(involving great attention of details) with less choopy(having a disjoined or jerky quality) and smoother for subregion choice.**

###### **These m's(m_1.........m_n) are then pass through a softmax fn which outputs them as probabilities(s_1.........s_n).<br>Finally, we take the inner product of probability vector(S) and subregions(y) to get the final output(Z), the relevant region of the entire image.**<br><img src="images/11.1.PNG" height="300" width="1000">
> **`Understand the probabilities as correspond to the relevance of the subregions(y) given the context(C).``???`<br>**

**Types of Attention**<br> 
**1. Soft Attention:**  *[The main relevant region(z) consists of different parts of different sub-regions(y)]*<br><img src="images/soft_att.PNG">
> ###### **`Z = sum(s_n.y_n)`    where, s=probabilities of the sub-regions(y).
###### **Soft Attention is deterministic. What "deterministic" means?- A system is said to be deterministic if the application of an action(a), on a state(s), always leads to te same state(s')<br>For example, Suppose, you face to the forward standing in the corner of a room and you then one step ahead 5 feet to the forward. Now you have a new state with coordinates(5,0) and stil facing to the forward. That changes your location(coordinates) but your state remains same. Every time you try this, the state will be same, hence it is deterministic.**
<img src="images/14.PNG" height="300" width="1000">

**Lets apply the same concept of soft-attention**
###### **Initially, we have an image split into a number of regions(y_i) with an input context(C). This is our initial state. On the application of soft detection we end up with a localized image representing the new state(S'). These regions of interest are determined from (Z). The output will always be the same regardless of how many times we execute self-attention with these same inputs. This is because we consider all the regions(Y) anyways to determine (Z).**<br><img src="images/15.PNG" height="300" width="1000">
**2. Hard Attention:**  *[The main relevant region(z) consist of only one of the regions(y)]*<br><img src="images/hard_att.PNG">
###### **Instead of taking weighted arithmetic mean of all regions, hard attention only consider one region randomly. So, hard attention is a `stochastic` process.<br>What `stochastic`(Randomness) mean?- Performing an action(a), on a state(S) may lead to different states every time.**<br><img src="images/16.PNG" height="300" width="1000">
###### **`What makes Hard Attention stochastic?-` is that a region(y_i) is chosen randomly with the probability(s_i). The more relevant a region(y_i), as-a-whole is relevant to the context, then grater the chance it is chosen for determinig the next word of the caption.**<br><img src="images/17.PNG" height="300" width="1000">

**Attention(Cont..)**
###### **Using the word captions output until now by the RNN, that is (H), along the current regions of interest in an image determined by the attention mechanism, the RNN now tries to predict the next word in the caption.**<br><img src="images/18.PNG" height="300" width="1000">

**Performance**
* ###### **Performance varies on dataset.**
* ###### **Hard attention perform slightly better while soft-attention perform decently well.**
**Application**
* ###### **`Neural Machine Translation (translate one language to another)` - Words are fed in a sequence to an encoder one after another and the sentence in terminated by specific input word or symbol. Once complete, the special signal initiates the decoder phase where the translated words are generated.**
* ###### **`Teaching Machines to Read`**