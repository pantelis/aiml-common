# News Time Machine

```{eval-rst}
.. youtube:: wArETCVkS4g
```

The video above helps you understand the topic of this project. 

You will be working individually to develop the services needed by a django-based recommendation system that will produce recommendations of a selected story across an archive of the .

In this case, the system will be 

1. Accepting as input youtube video transcriptions associated with instructive videos relevant to a category or role. Suggested roles are car mechanics, do it yourself and other instructive categories including those of education (Khan academy). 
2. Segmenting and converting the video into a suitable vector representation. Segmentation may include both the task / question and the answer.
3. Retrieve the most relevant documents that are stored in a database. The document is specific to the question that is asked. 
4. 
and presents them as posts / articles. You can select one partner to work with you on the project. Google News and techmeme.com are examples of such sites.

## Django site operation and Integration

The deliverable is the django web site that will display the news and the integration of the components.

### Milestone 1: Django site (10 points)

Learn one of the most popular web frameworks in the world, Django. You can start with the [official tutorial](https://docs.djangoproject.com/en/5.0/intro/tutorial01/) but bear in mind that you will also need to consult [Django Ninja](https://django-ninja.dev/) as **this will be the framework that you will use to build the API/App**.

You deliver a directory in your existing assignments repo called `django-news` containing the docker compose based deployment of django ninja with postgres, redis and celery. You can borrow the structure of the project from running [this cookiecutter](https://github.com/cookiecutter/cookiecutter-django) but ensure that the project is using Django Ninja.

### Milestone 2: Data Model (10 points)

Build a data model that will satisfy the needs of the project by making use of Pydantic 2 based data modeling tooling of Django Ninja. You can consult a slightly relevant to this exercise  [implementation](https://realpython.com/build-a-content-aggregator-python/) or any other implementation you think its relevant.  Note that the data model is for podcasts and you need to have a quite different data model than this. 

Deliver the data model as code (obviously), as a screenshot of the admin interface of the django site and as a screenshot of [DBeaver CE](https://dbeaver.com/docs/dbeaver/Database-Structure-Diagrams/) entity relationship diagram.

### Milestone 3: YouTube News (10 points)

Using [this channel](https://www.youtube.com/channel/UCYfdidRxbB8Qhf0Nx7ioOYw) download the transcripts of 10 videos of your choice from all 9 categories that youtube categorizes their news content (Top stories, Sports, Entertainment, Science, Health, Business, Technology, National, and World). Choose diverse new videos in terms of content. 

Using the Jetstream instance define a stream called "youtube-news" and publish the transcriptions of the videos in the stream. 


The stream consumer(s) retrieve the messages (each message is a video) and selected video URLs, their transcripts and other relevant metadata such as youtube channel, publication date are in a database table and are exposed to django app via your data model. 

Your Django views in this app are based on [Bootstrap 5-based Tabler.io](https://tabler.io/preview) components.

### Milestone 4: API (10 points)

Write the elements of the code that will subscribe to Kafka events such as [this](https://github.com/addu390/django-kafka). Please note that you can implement Kafka via [Redpanda](https://redpanda.com/). 

1. Ensure that the site is operational and can integrate in the intelligent services.

## Recommendation System

1. You will be building a recommendation system that recommends transcribed video content to all of the twitter topic categories. You will select 10 Twitter users (real people, not organizations) and determine which of the youtube transcriptions you need to feed the app for each one of them.  Your recommendations will be based on their past retweets and the tweets of the people they follow so select people that are very active on twitter. You can also select to target Twitter communities instead of individuals.  

2. The output should be published into kafka topics and stored in an Elastic Search server. 
   

## Transcription Retrieval and Summarization

1. You will be using the youtube data api to retrieve the transcriptions from each of the videos in the stream. 

2. You will store the transcription and summarize it using existing NLP models for text summarization. 

3. You will publish the transcription and summary in Kafka topics.  

## Semantic Search

1. You will be subscribe to transcription Kafka topics and  encode each transcription summary into a vector (embedding). See [this artcle](https://medium.com/version-1/vector-based-semantic-search-using-elasticsearch-48d7167b38f5) on how to do that in Elastic Search. Please note that as of the last few years, OpenSearch (the fork of ElasticSearch) has become the defacto standard for semantic search.

2. You will be offering via the django app the ability to the users to perform the semantic search and retrieve relevant transcriptions to keyword queries.




