# Webassembly (WASM) media pipelines

![](images/wasm.png)

This is a project for those that want to work on infrastructure / framework level code. The goal is to create a framework for running machine learning pipelines using webassembly components. This will allow us to run perception machine learning in the browser or in the cloud without tight coupling with specific language runtimes. 

The value proposition of this project: 

* pipelines can run in the client browser preserving privacy when applications cannot transmit user data to the cloud. 
* WASM runtimes are secure by design as they run in a sandbox. 
* pipelines can run in the cloud using webassembly runtimes such as [wasmtime](https://wasmtime.dev/) and can support a variety of languages and environments including optionally GPU acceleration. 
* Ease of use: pipelines can be built using a visual editor and can be deployed to the cloud or to the browser.

```{note}
Please [note the relationship between MediaPipe and this project](https://www.it-jim.com/blog/mediapipe-intro/). Both projects are working on similar problems, having said that the gstreamer basis caters to string real-time media processing and inherits the NVIDIA Deepstream nodes and plugins that accelerate media processing on NVIDIA GPUs.
```

## Project Goals

To demonstrate media processing using wasm pipelines we will build a simple application that can take a video stream from a webcam and apply a machine learning model to detect objects in the video stream.  

The project will be broken down into the following tasks:

### Task 1A: Familization with Gstreamer

![](images/gstreamer-overview.png)

[Gstreamer](https://gstreamer.freedesktop.org/documentation/application-development/introduction/gstreamer.html?gi-language=c) is an open source pipeline-based cross-platform extensible multimedia framework and has been adopted by major players in mission-critical use cases that need to process video and other ream-time streams. It also underpins [NVIDIA's DeepStream](https://developer.nvidia.com/deepstream-sdk) implementation. 


To get started in [Gstreamer](https://gstreamer.freedesktop.org/) please watch  **[this](https://www.youtube.com/watch?v=ZphadMGufY8)** video. [This additional](https://www.youtube.com/watch?v=_yU1kfcC6rY) hands-on addresses NVIDIA pipeline elements. **You do not need to have access to an NVIDIA GPU to work on this project.**

### Task 1B: Dear ImGui Bundle UI

The Gstreamer pipeline will be configured and controlled by [the Dear ImGui Bundle UI](https://code-ballads.net/annoucing-dear-imgui-bundle/) and extension of the Dear ImGui [that is heavily used](https://github.com/ocornut/imgui/wiki/Software-using-dear-imgui) for games and heavyweight computational applications such as [Blender](https://www.blender.org/) and NVIDIA Omniverse.

For a browser experioenec see [this demo](https://traineq.org/ImGuiBundle/emscripten/bin/demo_imgui_bundle.html). 

More specifically we need the capabilities of the Node Editor that allows gstreamer pipelines to be specified in the browser visually **and** for the user to be able to test the pipeline in the browser or using remote processing nodes in the cloud.

![](images/node-gui.png)

```{note}
The Dear ImGui Bundle python SDK allows the UI to be coded in Python. 
```

### Task 2A: Familiarization with WebAssembly

The implementation is being done here  https://fluendo.com/en/blog/gstwasm-gstreamer-for-the-web/ and the effort is described in [this video](https://gstconf.ubicast.tv/videos/gstwasm-gstreamer-for-the-web/). 

Webassembly runtimes such as [wasmtime](https://wasmtime.dev/) or serverless platforms such as  [cloudflare workers](https://developers.cloudflare.com/workers/runtime-apis/webassembly/) or [AWS Lambda](https://aws.amazon.com/lambda/) can be used. 

This section will be updated to reflect the results of the prototype that help us define appropriate runtime for this project. 








