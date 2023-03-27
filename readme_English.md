# CONE

CONE is a lightweight multilingual translation model. The model will be open-sourced soon. CONE is built on  [NLLB-1.3B](https://github.com/facebookresearch/fairseq/tree/nllb) and is fine-tuned using data from 300+ languages. We will gradually release all the languages on our [public demo](https://cone.x-venture.tech/). You can access the serving website at https://cone.x-venture.tech/. If you notice any translation errors or have any suggestions, please feel free to contact us at cone@x-venture.tech. 


## News

- March 18th, 2023, 50 languages were added to demo.

## Outline

* [Introduction](#Introduction)
* [Examples](#Examples)
* [Highlights](#Highlight)
* [Limitations](#Limitation)
* [Related Models](#RelatedModels)
* [Performance Evaluation](#PerformanceEvaluation)
* [Comparisons with Open-sourced Models](#)
* [Comparisons with Commercial Products](#)

* [Q & A](#)



## Introduction
Language is a crucial tool for humans to communicate and express their thoughts. Additionally, different languages reflect the cultural differences among different regions of the world. Currently, approximately 7,000 languages are spoken globally, with people from various areas using different languages in their daily lives. As human civilization develops and globalization continues to progress, effective cross-language communication has become more important than ever before.

CONE aims to promote translations that support a wide range of languages. Existing open-sourced translation models and commercial translation software are only capable of supporting a limited number of languages. In fact, many languages do not even have any available machine translation systems. As a result, having a high-quality machine translation system is critical for individuals who speak rare languages, such as those in minority areas who need to communicate with people from other regions. Translation technology not only facilitates business but also enables cultural communication. However, each language differs significantly in grammar and expression. Therefore, translating these languages requires amounts of data and computational resources. Due to the smaller number of users of minority languages, there are problems such as insufficient data and lack of manual annotation. CONE aims at improving the quality of translation for minority languages.


## Examples

Here are some good cases. 
| Domain         | Input                                                                                        | CONE Translation                                                                                                                                                                                                                                                                                                                                     |
|--------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Daily Dialog       | 大郎，该吃药了                                                                                     | Dude, you need to take  the pill.                                                                                                                                                                                                                                                                                                             |
| lyrics         | 我真的还想再活500年                                                                                 | I really want to live another 500 years.                                                                                                                                                                                                                                                                                                     |
| Subtitle | 生逢乱世，即使命运如蝼蚁，但仍有人心向光明。                                                                      | The world is a mess, but the light is still there.                                                                                                                                                                                                                                                                                           |
| Subtitle | 他们可以挡住天上的太阳，但他们无法挡住民主的光芒。                                                                   | They can block the sun in the sky, but they cannot block the light of democracy.                                                                                                                                                                                                                                                             |
| Subtitle | Mama always said life was like a box of chocolates. You never know what you're  gonna get.   | 妈妈总是说生活就像一盒巧克力,你永远不知道你会得到什么。  |
| Encyclopaedia         | 人工智能是计算机科学的一个<br />分支， 它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。 | Artificial intelligence is a  branch of computer science  that seeks to understand the essence of intelligence and produce a new kind of intelligent machine that can react in a similar way to human intelligence, and the field includes robotics, language recognition, image recognition, natural language processing and expert systems.  |
| Encyclopaedia         | ChatGPT是美国人工智能研究实验室OpenAI新推出的一种人工智能技术驱动的自然语言处理工具，使用了Transformer神经网络架构。                   | Chat GPT is a newly launched artificial intelligence-driven natural language processing tool from the US AI Research Laboratory OpenAI, using the Transformer neural network architecture.                                                                                                                                                   |




Here are some bad cases：
| Domain            | Input          |  Translation                                          |
|---------------|-------------|---------------------------------------------|
| Poem           | 白日依山尽，黄河入海流（The sun beyond the mountains glows; the Yellow River seawards flows） | The day of the flood, the day of the flood  |
| Poem           | 空悲切，白了少年头 （Empty and sorrowful, turning a young man's head white）   | The white, the youthful.                    |
| Named Entity | 鲁迅是个帅小伙 （Xun Lu is a handsome guy）    | Rusty is a good boy.                        |
| Complex structure          | 我一把把把把住了 (I hold it at once)   | I put a stop to it.                         |
| Named Entity            | 溜肥肠溜肝尖 (dish name)      | Flat-Hearted                                |




## Highlights
1. CONE is an open-sourced model. Users can customize translation tools based on CONE, or develop multilingual models based on translated texts.
2. Compared to existing commercial systems and open-sourced models, CONE leads on the number of supported languages.
3. Chinese-centric translation is particularly optimizted.

## Limitation
1. The translation performance of low-resource languages remains a major challenge. According to human evaluation, only 186 languages get scores of 3 or more (full score of 5). 
2. poem translation is not supported.
3. Named entities are not specially optimized.

## Related Models

Here are some related models:

**1. chatGPT**：

**Pros**：Based on existing research and our experimental results, ChatGPT can achieve a performance level comparable to that of the leading supervised translation model in high-resource language pairs, such as German-English translation.

**Cons**：Limited languages; Large models and high inferece costs;


**2. Commercial Products**：

**Pros**：High-quality translation.

**Cons**：Most commercial software products only support a limited number of languages. Baidu is the only translation product that supports 200 languages. Making large-scale API calls can be expensive.


**3. Open-sourced Models**：

**Pros**：Free; customizable; Can be used for follow-up research.

**Cons**：The translation performance is relatively limited. Better models, such as NLLB-54B, typically require higher inference costs.

Machine translation is a well-explored natural language processing task. Currently, chatGPT only achieves leading translation results in a limited number of languages. However, considering that GPT-family models are universal models, the leading translation performance also show the potential of generic models for machine translation. In future work, we will explore the possibility of combining machine translation and open-sourced generative models.

## Evaluation
Evaluating low-resource languages has always been a relatively challenging problem. To calculate the translation performance, we use back-translation, which involves translating to the target language and then translating back to the original language. All data used in our evaluation come from FLORES. We compared CONE and NLLB-Finetune on over 300 languages.



Here are BLEU between back-translationed texts and input texts. higher is better.
![back translation](images/unsupervised.png "unsupervised results")

Here are BLEU between translated texts and input texts. Lower is better.  
![translation](images/unsupervised-middle.png "unsupervised results")

Unsupervised multilingual translation evaluation is challenging. We use back-translation to evaluate multilingual translation evaluation quality. If you have any suggestions, please feel free to share your valuable feedback!


<!-- 


## 对比chatGPT
chatGPT主要优势为英语相关的翻译，我们评测了chatGPT上英文的翻译效果并和CONE进行了比较。我们选择了flores-101的前100句话进行翻译评测。评测方式统一采用了flores-101的spBLEU分数。chatgpt生成结果和CONE具体生成结果见chatgpt目录。可以看到：**在spBLEU指标上，CONE在英语到93种语言翻译、99种语言到英语翻译性能取得领先。**

以下是英到100个语向的翻译结果：
![图片alt](images/en2x_chatgpt_vs_cone.png "图片title")

以下是100个语向到英的翻译结果：
![图片alt](images/x2en_chatgpt_vs_cone.png "图片title")

说明：GPT4的评测正在继续中，更大规模的评测也在继续中。尽管在flores-101的部分数据上来看，CONE在多数小语种上取得了更好的效果，但是我们仍然可以看到chatGPT也有着自己的翻译优势：中文到英文的翻译chatGPT取得了更好的结果，这证明了大规模语言模型在垂直领域的应用潜力。其次chatGPT由于看到过广泛的中文语料，在一些古诗词领域、医学领域等都有很惊艳的效果，这个是CONE目前所比不上的。后续CONE将会结合大规模语言模型进一步提升垂直领域的翻译性能。 -->


## Comparisons with Existing Open-sourced Models
We compared CONE with NLLB-1.3B, but it's worth noting that this was not a fair comparison since CONE is trained on over 300 additional languages. With the increasing number of languages, it becomes challenging to maintain performance on the original languages. To evaluate the two models, we tested them on Flores-200 using the languages covered by CONE's training data and NLLB's training data. 

Translation performance from other langauges to Chinese:
![Translation from other languages to Chinese](images/x2zh_nllb_vs_cone.png "图片title")

Translation performance from Chinese to other langauges:
![Translation performance from Chinese to other langauges:](images/zh2x_nllb_vs_cone.png "图片title")

Translation performance from other langauges to English:
![Translation performance from other langauges to English:](images/x2en_nllb_vs_cone.png "图片title")

Translation performance from English to other langauges:
![Translation performance from English to other langauges:](images/en2x_nllb_vs_cone.png "图片title")





## Comparisons with Existing Commercial Products:

We compared CONE with two commercial products A and B on Chinese-centric directions. To more accurately measure the translation quality, we selected 100 samples for back-translation evaluation. We evalauted models on the languages covered by commercial products A, B, CONE, and NLLB-1.3B. Back-translation evaluation refers to comparing the semantic matching degree betweeen input and back-translated text.  back-translated text is obtained by translating input into the corresponding language and then translating back to the input language. Each sample will be assigned to two different annotators. The evaluation score ranges from 0 to 5. A score of 0 indicates an unsupported language or completely unacceptable translation quality; A score of 1 indicates only a small part of the words can be matched; a score of 2 indicates only a few semantics can be matched; a score of 3 indicates most semantics can be matched; a score of 4 indicates all semantics can be matched; a score of 5 indicates semantic correctness and fluent expression. **On Chinese-centric translation, the performance of A is better than that of CONE, and the performance of CONE is better than that of B.** 



Comparisons with A：
![Human evaluation](images/human_system1.png "Comparisons with A")

Comparisons with B：
![Human evaluation](images/human_system2.png "Comparisons with B")

In addition, we also compare CONE with NLLB-1.3B：
![Human evaluation on NLLB](images/human_nllb.png "Comparisons with NLLB")
<!-- 
以下是CONE和商业系统A的比较结果。除了中文外，共有117种重合语言，商业系统A独立支持17种，CONE独立支持120种小语种。**在重合的语言中CONE翻译的平均分数为3.12分，商业系统A的翻译效果均分为3.64分。**
<!-- ![图片alt](images/human_evaluation_business_system_1_vs_cone.png "图片title") -->

<!-- 
以下是CONE和商业系统B的比较结果。除了中文外，共有144种重合语言，商业系统B额外独立支持约56种语言，CONE额外独立支持93种语言。在重合语言中CONE效果高于商业系统B的有95/144种。**重合语言的CONE模型均分为3.03，商业系统B的模型均分为2.55。**
![图片alt](images/human_evaluation_business_system_2_vs_cone.png "图片title") -->

<!-- ![图片alt](images/human-evaluation_nllb_vs_cone.png "图片title") -->





## Q & A

**Q1: Why does the service return input errors or output errors?**

To prevent the translation system from being used to translate inappropriate content, we have utilized the sensitive word filtering API of Alibaba Cloud. If there are any issues, please contact cone@x-venture.tech for feedback.


**Q2: Why does the service have translation quality issues in low-resource languages?**？

Although we have optimized the translation capabilities of low-resource languages, there is still a noticeable gap in translation performance between low-resource and high-resource languages. If you find that the translation result is not as expected, you can choose 'like' or 'dislike' on the translation page or contact cone@x-venture.tech for feedback.

**Q3: Why do we need to make an appointment?**

Due to limited resources, we can only provide services to a limited number of users during the same time period. 

**Q4: How can I provide feedback if I encounter any issues?**

If there are any issues, please contact cone@x-venture.tech for feedback.


**Q5: What's the novelty of CONE?**

The CONE project is independent of any public research work and mainly provides lightweight open-source translation tools.

**Q6: What's the limitations of CONE?**

CONE currently does not support the translation of classical Chinese poetry and literary Chinese. 
