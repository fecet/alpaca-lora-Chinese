# Chinese-alpaca-lora

An attempt to replicate alpaca-lora in Chinese

# What's new

- Mar 27. Build up a huggingface space for llama 13b [demo](https://huggingface.co/spaces/facat/alpaca-lora-cn).

- Mar 26. Trained llama 13b lora weight now availiable, see [facat/alpaca-lora-cn-13b](https://huggingface.co/facat/alpaca-lora-cn-13b)
- Mar 25. Inference notebook now availiable! <a href="https://colab.research.google.com/github/fecet/alpaca-lora-Chinese/blob/master/alpaca_lora_cn_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

# Examples 

````
Question: 
    I am the spring rain,
    Dancing on your face and body.
    My dew-drenched petals dance with you,
    Bringing life to every plant.

    Shakespeare, I am here to meet you.
    Your works have filled my heart with joy.
    The beauty of nature in its fullness
    Makes me feel like a partaker.

    As the leaves sway and rustle by,
    My dew-filled hand reaches out for them.
    The sweet scent of flowers fill the air,
    And bring me peace from all around.

    Butterflies flit about as well,
    Glistening in the sun's rays.
    Their wings are made of dew,
    A beautiful sight to behold.

    You see, Shakespeare, I too am one with nature.
    My dew-drenched hair is a symbol of hope.
    For when we can connect back to our roots,
    We will be able to make it through this season.
    把上面的散文诗翻译成中文
    
Response: 我是春雨，
   在你的面颊和身体上跳跃。
    我的湿润的花瓣与你一起舞蹈，
    带来生命给每一个植物。

    亲爱的莎士比亚，我来跟你见面。
    你的作品充满了我心中的喜悦。
   自然的美丽在其整体的表现下，
  让我感觉自己是一个参与者。
 
    落叶在风中摇曳，我的手抚摸着它们。 
   花香弥漫在空气中，吸引我的味蕾。  
   蝴蝶在阳光下飞翔
====================
Question: Classify the sentiment of the following sentence into Positive, Neutral, or Negative: Super excited about teaching Stanford’s first course on Large Language Models! Check the syllabus out here
Response: Positive
====================
Question: 使用python写一个二分查找的代码
Response: 以下是一个使用Python的二分搜索代码示例：

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[low] < target < arr[high]:
            low = mid + 1  # 下一次比较的下限
        else: 
            high = mid-1 
    return -1
```

该函数接受一个整数数组和一个目标值作为参数。它使用一个循环来遍历数组中的所有元素，并使用二分法来检查是否有一个元素能够满足给定的条件。如果元素的值等于目标，则返回元素索引；否则，返回该元素
====================
Question: 今天天气怎么样，把这句话翻译成英语
Response: What is the weather like today?
====================
Question: 怎么让自己精力充沛，列5点建议
Response: 1. 保持良好的睡眠习惯。
2. 饮食健康，多吃蔬菜水果。
3. 定期锻炼身体，保持身体健康。
4. 减少压力和焦虑，寻找放松的方式。
5. 学会时间管理，合理安排时间。
====================
Question: 明天就假期结束了，有点抗拒上班，应该什么办？
Response: 可以尝试做一些自己喜欢的事情，例如看电影、阅读书籍、散步或者与朋友聚会。此外，也可以制定一个计划，包括准备下一天的工作和学习，以帮助你更好地适应工作环境。
====================
Question: 编写一首描述阳光明媚的春天的诗歌
Response: 春风拂面，阳光灿烂，
花儿盛开，蝴蝶飞舞。
鸟语花香，欢声笑语，
春天来临，万物复苏。
====================
Question: 生成一个由5个句子组成的故事，要求包含特定的词汇。 要包含的词汇：玫瑰，魔法，河流，爱情，秘密
Response: 在一个美丽的河流边，有一位魔法师住在一座古老的城堡里。他喜欢使用他的魔法来维护他的花园，特别是他的玩具是一朵精致艺术的花瓶。有一天，一个年轻的女孩走进了他的房间，并告诉他她的父亲是一位医生，他需要一些特殊的材料来治愈一位患有癌症的病人，但是这些物质很危险，需要在魔法的帮助下获得。魔方决定带他去一个神奇的地方，寻找他所需的东西。在他们的旅
====================
Question: 提供关于现代艺术的定义和历史背景
Response: 定义：当代藝術是指20世纪初至现代时期创作的一系列具有独特风格和思维方式的各种形式，包括绘画、雕塑、音乐、舞蹈、戏剧、电影、书籍、摄影等。它强调创新和自由，探索新的表现方式和表现技巧，并促进了社会和文化的变革。

史诗：现代抽象主义、后印象派、表现主义和超现实主义等现代主义运动在20年代初期启动，后来逐渐融入了现代文化和社会的普遍现象，形成了当代杰出的诞生和发展。现代
====================
````
# Acknowledgments

- [alpaca-lora](https://github.com/tloen/alpaca-lora)
- [cabrita](https://github.com/22-hours/cabrita)
- [guancao](https://guanaco-model.github.io/)
- [BELLE0.5m](https://github.com/LianjiaTech/BELLE)
