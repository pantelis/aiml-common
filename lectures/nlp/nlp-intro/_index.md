---
title: Introduction to NLP
weight: 131
draft: false
---

# Introduction to NLP

## The Natural Language Understanding  Agent

![nlp-architecture](images/nlp-architecture.png#center)
*Natural Language Understanding as an AI Agent that offers language comprehension and logical reasoning. NLU is an unrealized dream today but elements of NLU are being actively researched as we speak.* 

NOTE: These notes will be supplemented with an explanation of the transformer architecture. The attention based approaches revolutionized the NLP field and the reader that wants to deep dive into transformers will be rewarded by the treatment [here](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

## Natural Language Processing (NLP) 

The much simpler NLP task is a selective subset of all functionalities of the NLU architecture above.  To get started, we need some common ground on the NLP terminology - the terms are presented in the processing order of an NLP pipeline \[[cite](https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e)\].

| Term | Definition |
| --- | --- |
| **Segmentation** | The first step in the pipeline is to break the text apart into separate sentences. Coding a Sentence Segmentation model can be as simple as splitting apart sentences whenever you see a punctuation mark. But modern NLP pipelines often use more complex techniques that work even when a document isn’t formatted cleanly. Take for example the sentence “London is the capital and most populous city of England and the United Kingdom.” |
| **Word Tokenization** | The next step in our pipeline is to break this sentence into separate words or tokens. This is called tokenization. “London”, “is”, “ the”, “capital”, “and”, “most”, “populous”, “city”, “of”, “England”, “and”, “the”, “United”, “Kingdom”, “.” |
| **Part of Speech Tagging** | The next step tags each word with a classification result of its part - whether it is a noun, a verb, an adjective and so on. We can do this by feeding each word (and some extra words around it for context) into a pre-trained part-of-speech classification model:|
| | ![pos-classification](images/pos-classification.png#center) |
| | ![pos-example](images/pos-example.png#center) | 
| **Lematization** | This is one of the stages of _vocabulary normalization_.  At this stage we are figuring out the most basic form or lemma of each word in the sentence. |
| | ![lemmatization](images/lemmatization.png#center) |
| **Stop words identification** | This is one of the stages of _vocabulary normalization_.  Stop words are common words in any language that occur with a high frequency but carry much less substantive information about the meaning of a phrase. |
| | ![stop-words](images/stop-words.png#center) |
| **Dependency Parsing** | The next step is to figure out how all the words in our sentence relate to each other. This is called dependency parsing. The goal is to build a tree that assigns a single parent word to each word in the sentence. The root of the tree will be the main verb in the sentence. In addition to identifying the parent word of each word, we can also predict the type of relationship that exists between those parent and child. This parse tree shows us that the subject of the sentence is the noun “London” and it has a “be” relationship with “capital”. We finally know something useful — London is a capital. Dependency parsing in NLP stacks circa 2017 was heavily based on [deep learning](https://ai.googleblog.com/2017/03/an-upgrade-to-syntaxnet-new-models-and.html) architectures - [demo](https://explosion.ai/demos/displacy?text=the%20final%20exam%20will%20be%20a%20complete%20bloodbath&model=en_core_web_sm&cpu=1&cph=1) |
| | ![dependency-parsing](images/dependency-parsing.png#center) |
| **Named Entity Recognition (NER)** | The goal of Named Entity Recognition, or NER, is to detect and label nouns with the real-world concepts that they represent. NER systems aren’t just doing a simple dictionary lookup. Instead, they are using the context of how a word appears in the sentence and a statistical model to guess which type of noun a word represents. [demo](https://explosion.ai/demos/displacy-ent?text=The%20Supreme%20Court%20on%20Thursday%20ruled%20that%20Manhattan%27s%20chief%20prosecutor%20can%20obtain%20troves%20of%20President%20Trump%27s%20business%20records%20and%20tax%20returns%2C%20a%20momentous%20defeat%20for%20the%20president%20in%20his%20efforts%20to%20shield%20his%20personal%20financial%20information%20from%20state%20investigators.%0A%0A%0AThe%20high%20court%20ruled%207-2%20in%20favor%20of%20Manhattan%20District%20Attorney%20Cyrus%20Vance%2C%20who%20is%20conducting%20a%20criminal%20investigation%20into%20the%20president%27s%20business%20dealings%20and%20hush-money%20payments%20made%20to%20two%20women%20who%20allegedly%20had%20affairs%20with%20the%20president%20years%20before%20he%20was%20elected.%20Justices%20Neil%20Gorsuch%20and%20Brett%20Kavanaugh%2C%20appointed%20to%20the%20high%20court%20by%20Mr.%20Trump%2C%20joined%20the%20majority%2C%20while%20Justices%20Clarence%20Thomas%20and%20Samuel%20Alito%20dissented.%20The%20justices%20sent%20the%20dispute%20back%20to%20the%20lower%20courts%20for%20further%20proceedings.&model=en_core_web_sm&ents=person%2Corg%2Cgpe%2Cloc%2Cproduct%2Cnorp%2Cdate%2Cper%2Cmisc)|
| **Coreference Resolution** | Coreference resolution is the task of finding all expressions that refer to the same entity in a text. It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction. Coreference resolution is one of the most difficult steps in our pipeline to [implement](https://huggingface.co/coref/) |
| | ![coreference-resolution](images/coreference-resolution.png#center) |


Python and almost all programming languages are formal. They define a strict set of rules called a _grammar_ that the programmer must follow religiously. In addition formal languages also define semantics or meaning of the program via a set of rules. So the exit code ```0``` after the execution of a routine has the meaning of ```successful termination``` but a ```1``` the opposite. The English language on the other hand is "free wheeling" so to speak due to its ambiguity ("he met his maker") and size. 

{{< columns >}} 
## Same context, different meaning
"The food in this restaurant was good, not bad at all"

"The food in this restaurant was bad, not good at all."
<---> 

## Different context alltogether 
"The bank's profits eroded soon after the 2008 crisis"

"The river's bank eroded after the 2008 floods"
{{< /columns >}}


To capture the fuzziness of natural language we define language models probabilistically. We do not speak of a single meaning for a sentence - rather we speak of _a probability distribution over possible meanings_. In the next section we will start to quantify such probabilistic language models in detail. 


