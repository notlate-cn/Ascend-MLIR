<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0108.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# TPipe简介

# TPipe简介

TPipe用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。其主要功能包括：

  * **内存资源管理** ：通过TPipe的InitBuffer接口，可以为TQue和TBuf分配内存，分别用于队列的内存初始化和临时变量内存的初始化。
  * **同步事件管理** ：通过TPipe的AllocEventID、ReleaseEventID等接口，可以申请和释放事件ID，用于同步控制。



**父主题：** [TPipe](atlasascendc_api_07_0108.html)
