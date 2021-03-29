---
title: Collaborative Robotics
---

# Collaborative Robotics

You are a co-founder in a startup called PodGrocer, a company that deploys pods to selected host sites such as electric vehicle charging stations, mall parking lots and can also attach to existing supermarket stores. Each pod was constructed by modifying a standard shipping container and is cooled by latest technology green energy backed up by the host site power feeds. 

{{<hint info>}}
The shipping container sized pod is irrelevant for this project but its provided here as they are routinely used as building blocks for larger and sometimes movable structures. 
{{</hint>}}

The system in operation is shown in the following video

{{<youtube n5G3KfE8PVc>}}

How the company generates revenue: 

1. Customers order via PodGrocer's or partners web site. Partner's fulfillment center places the order in a standard container called a tote (blue container). Multiple totes populate a palette that can be moved around by a mobile robot. 

2. The pod is replenished by a truck delivery driver (or fulfillment center personnel if the pod is attached to an existing facility). They roll the palette in via the pod's loading bay and from that point onwards, everything is automated. 

3. Customers that orders groceries and other goods, pick up their order via an ATM-like hutch.  The customer's app provides an estimate as to when the customer is expected to be at the pod location. After arrival and authentication, the system is able to deliver the relevant to the order totes to the customer at the pick up hatch in seconds. 

See [this video](https://www.youtube.com/watch?v=IqYk0dFcZgc&t=2s) for more details of the robotic pod operations. 

## Tasks

In this project you will perform certain tasks to ensure that the pod is able to perform its required robotic functionality and function as a system. The functionality is divided into four subsystems: 

1. Path Planning Controller responsible for executing path planning. 
2. Navigation Controller responsible for localization. 
3. Logistics Controller
4. Integration Controller. It may optionally present as it depends on your design and implementation. If it is present, it may delegated the role of an orchestrator responsible for running the demo of the whole system.

The task you need to do is to implement and use the above subsystems to coordinate the transfer of pallets inside a pod. 

Task 1: (25 points) Create a collision free plan. No robots should collide with each other. This task involves controllers 1 and 4. 

Task 2: (75 points) Move at least 2 pallets from pallet locations to storage location with the first attempt. This task involves controllers 1, 2 and 4. 

Task 3: (Extra 20 points) Simulate a demand distribution of temporal arrival rate of customer orders characterized by a random number of totes per order up top 10 totes per order and assume a customer arrival pickup time move the pallets according to the packing efficiency 

Implementation-wise you can follow Track 1 or Track 2 versions of the project below. The reason why we have two tracks is to allow students that are remote and unable to participate into teams to win 100 points without having dependencies on ROS and relatively extensive compute environments. There is no free lunch for Track 2 though as some of the functionality that is readily available in Track 1 needs to ve developed in track2.  

To help plan the tasks amongst the team members please use Trello (login with your nyu account) and split the project into four boards one for each controller above. Also 

{{<hint alert>}}
Note that the points are assigned per task completion and are independent on how many controllers you implemented. Therefore make the controllers initially simple and make sure you submit both completed tasks rather than spending time on the ultimate path planner and miss the deadline.  
{{</hint>}}

## Track 1 (ROS Foxy)

The implementation must be resilient and therefore the system must be distributed across many nodes supported by the ROS as well as use a number of AWS managed services. Usage of AWS managed services are not required for this project.  

The Robot Operating System (ROS) is a set of software libraries and tools for building robot applications. From drivers to state-of-the-art algorithms, and with powerful developer tools, ROS has what you need for your next robotics project. And it’s all open source. [Here](https://docs.ros.org/en/foxy/index.html) you will find the official documentation on ROS 2, the newest version of ROS that you will use.


### Path Planning Controller

See the Dock-Worker Robots Planning Domain and the [Improving Classical AI Planning Complexity](https://towardsdatascience.com/improving-classical-ai-planning-complexity-with-planning-graph-c63d47f87018)


### Navigation Subsystem

For an introduction to the navigation system, see the following video:

{{<youtube QB7lOKp3ZDQ>}}

![architectural-diagram](images/architectural-diagram.png)

Localization: https://github.com/SteveMacenski/slam_toolbox


### Logistics Controller


## Track 2 (Python Robotics)

The implementation must demonstrate the tasks using the Python Robotics or any other Python library you need to. 

{{<hint info>}}
You need to assume the exact same environment as given by the map of Track #1 i.e. same dimensions and the same layout of warehouse (pod), same robot physical dimensions (rather than a single point), same dimensions for each palette etc. In addition you need to block specific occupancy maps. 
{{</hint>}}


### Motion Planning Controller


### Navigation Subsystem

You are welcomed to try various navigation libraries but you probably need [this](https://github.com/splintered-reality/py_trees) to match the flexibility of BTs used in Nav2 stack in Track#1. 


### Logistics Controller

