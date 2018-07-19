# Building a GAN without mode collapse

Please look at testGAN.ipynb to see my results

## How to eliminate mode collapse

When doing my research on GANs and their shortcomings, the main issue that came up was mode collapse. In order to properly start working with real data I wanted to produce, I needed to first discover how to eliminate possible issues with mode collapse.

### What is mode collapse?

Mode collapse occurs when a generator produces realistic data, but only of a certain type. Take for example the mnist database. We would want our generator to produce samples like this:

![alt text](https://www.researchgate.net/publication/298724407/figure/fig4/AS:340629608517635@1458223835091/Part-of-samples-of-MNIST-database.png)

However, the generator produces let's say this:

![alt text](https://i.ytimg.com/vi/ktxhiKhWoEE/hqdefault.jpg)

The generator has lost a few modes(in fact most of them). The issue is because it can make realistic 1s, it does not have any reason to produce anything else unless the disciminator forces it to. Although this seems like a small issue, it is in fact one of the biggest issues with GANs today. I have read at least 5 papers that have come out within the last year all trying to solve this issue.

### My experiment

My experiment was fairly simple, try and reproduce a simple 8 mode ring around the unit circle.

### How did I solve mode collapse?

I used two papers to solve the issue: D2GAN and PacGAN. Their papers can be found here https://arxiv.org/abs/1709.03831 and here https://arxiv.org/abs/1712.04086. I also considered using VEEGAN and Unrolled GAN but I will get to them later. The ideas behind both D2GAN and PacGAN are quite simple. In PacGAN we feed the discriminator two batches at a time. A simple idea, but, as seen in the paper, an effective one. In fact the paper showed the best performance for eliminating mode collapse out of any of the alternatives tested. Honestly, when I did my first simulation I had assumed that PacGAN would work by itself, especially with such simple data. Unfortunately it only added one or two modes to its output. 

I then tried also adding the D2GAN, which adds a second discriminator. With a second judge, the generator would not be pulled towards one mode, and need to create diverse enough data to fool both discriminators. Also a simple idea with good results. A big reason I picked these two methods was due to their simple implementation combined with their very good results. With the added boost of the D2GAN, the generator hopefully would improve. It did improve, but still missed some modes. In fact I reran this simulation around 100 times with modified parameters without much success. 

The epiphany actually occurred when looking closer at the discriminator after for a few hundred epochs pretraining it. It was reaching over 90% accurracy. This at first made me think I would need to stop pretraining it since it was making it too difficult for the generator to catch up. However, when I removed pretraining, the discriminator very quickly discovered what was real and what was fake. After printing initial generator outputs with some finetuning, the generator either produced output way too far outside the unit circle, or way too far inside. I adjusted the parameters to be within a unit circle of around 1.5-2 units, and suddenly mode collapse was gone. In fact the generator learned super quickly. 

This makes sense especially with real life analogies. If I was trying to cook and I was using vegetables and meats, someone could quickly tell me various recipes I could learn with many combinations. However, if I was just pulling random things out of a field like grass and dirt, someone would just be very confused on how to help me. They would most likely say something like, "Find a potato and we will bake it." I would become good at cooking potatoes, but the judge would also realize that I can only cook potatoes and criticize me. I will then find another vegetable to cook and cook just that. Starting in a good place is important.

I also made my network be a bit more complex with more hidden layers and nodes and the machine learned very quickly. I realized all my previous issues were most likely due to this impractical starting data.

### Other possible solutions

I mentioned VEEGAN (https://arxiv.org/pdf/1705.07761.pdf) and Unrolled GAN (https://arxiv.org/abs/1611.02163) in my proposed methods. The main reason I did not implement them is because at first, they seemed to have a much higher difficulty to results ratio. I will go into how they work though.

VEEGAN works by training a third network, known as the reconstructor, which will invert the generators output to a gaussian. It also does this with the true data. By comparing the two gaussian distributions, we can easily detect mode collapse. In order to train the three networks, the reconstructor tries to learn the generated and true data's conversion to a gaussian simultaneously. We train the generator and reconstructor through an objective function that essentially measures the difference between their gaussians. I did not spend a lot of time on this paper since PacGAN performed better and I am still self-learning stats, so some of the concepts were a bit foreign to me.

Unrolled GAN works by having the generator "see" into the future of what the discriminator will do. The idea is that instead of optimizing against what the discriminator is saying now, let's try and optimize against what the discriminator would say if it took 2+ steps instead. In many mode collapse cases, the issue is that the discriminator becomes too good, making it hard for the generator to learn. By giving the generator a bit of an edge, we hopefully can reduce the mode collapse. Unfortunately, this means calculating gradients a factor of 2+ times more often, which gives you less speed. Also, from the implementations I saw online, there were often hacks involved to make it work with the current codebase(at least in tensorflow). Since PacGAN worked better, I chose not to deal with these issues.

## Future work

The main reason I am testing this is because I want to reproduce realistic time samples of network traces. I want them to be realistic enough, and cover all the modes, so when I use this synthetic data to train another neural network, it will be learning based on how the real traces are. This involves using these techniques in a RNN setting. Current papers I'm looking at are seqGAN and stepGAN. 

These experiments also made it very clear to me that the starting point of the generator is very important. This is not something I was able to find online as a tip for making GANs without mode collapse. In fact, it seems really intangible what a good starting point would be for something with less concrete modes like cat images, or even numbers. I was considering, perhaps it is worth training our GAN to initially have as realistic as possible samples first. I will try considering how we can look at random outputs of a GAN, and then modifying the weight parameters to be at least acceptably close through regressive learning. When working with even more complex generations, I would assume that these kinds of initial setups are probably needed to see good results. Instead of me mindlessly twiddling with the starting parameters, we could learn how to setup properly. 
