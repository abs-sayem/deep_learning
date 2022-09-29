### **Attention Neural Network**
**What is Attention?**
###### **Suppose I give you `Deep Learning with Python` book and ask you to tell about `neural style transfer` from the book. What would you do - either start reading the whole book or check the index to find out where "neural style transfer" is talked about, go there and read the related portion. The second way is more precise because you draw the attention in the region of interest.**
**Understand Attention Mechanism**
###### **To understand attention mechanism here we will consider `Image Captioning through Attention`. How we can caption an image? The basic idea is- We will `Encode` the input image by a `CNN` then `Decode` the encoded representation by a `RNN(say-LSTM)` into word vector(ie. Caption).**
###### **`Image`--(Encoded by CNN)-->`H`--(Decoded by LSTM)-->`Caption`; where H=internal vector representation, Caption=into word vector.**
**[NB]`The problem is for captioning the image LSTM consider the entire image vector representation(H). This is not an efficient way, because- we actually caption a specific region not the whole image.**