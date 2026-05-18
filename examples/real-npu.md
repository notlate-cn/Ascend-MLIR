# 真实NPU环境和使用指导

## 1. SSH连接host信息
公网IP：<real-npu-host>
端口号：141
用户名：root
密码：<redacted-password>

## 2. Host工作目录和环境安装

### 2.1 工作目录
这是一台公共环境，只允许在 `/data/nyh` 目录下新增、修改、删除文件，务必谨记，务必谨记，务必谨记。

### 2.2 环境安装
若要使能NPU硬件，需要安装其依赖的驱动包和软件包。

#### 2.2.1 驱动包安装
目前驱动包已经安装好了，可以通过命令`npu-smi info`查看NPU硬件状态，能够看到是Ascend 910C，共16张卡，我们只允许使用7卡(`export ASCEND_DEVICE_ID=7`)。其安装在系统目录/usr/local/Ascend/driver目录下，后续每次建立shell会话，先执行命令`source /usr/local/Ascend/driver/bin/setenv.bash`，设置driver相关环境变量。

#### 2.2.2 软件包安装
软件包(toolkit工具包)只能安装到每位使用者的独立目录，比如我们的话就是第1点中的`/data/nyh/Ascend`目录下。

##### (1) 软件包下载
* 方式一：社区版（稳定但缺少最新功能），暂不使用此方法。
* 方式二：开发版（及时但不稳定），下载链接：https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/{时间戳}/ 。该目录下有x86和arm(aarch64)两种CPU架构的包。对于我们当前host环境(aarch64)，只下载安装两个包即可：
  * toolkit工具包：[Ascend-cann-toolkit_\${cann_version}_linux-\${arch}.run](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-toolkit_9.1.0_linux-aarch64.run)
  * ops算子包：[Ascend-cann-\${soc_name}-ops_\${cann_version}_linux-\${arch}.run](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run)

##### (2) 软件包安装
针对方式二开发版，安装2个包即可。
```shell
cd /data/nyh
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_9.1.0_linux-aarch64.run
# 安装指定目录 --install-path=/data/nyh/Ascend
./Ascend-cann-toolkit_9.1.0_linux-aarch64.run --full --quiet --install-path=/data/nyh/Ascend
./Ascend-cann-A3-ops_9.1.0_linux-aarch64.run --install --type=toolkit --quiet --install-path=/data/nyh/Ascend

# 本次实测 9.1.0 开发包生成的是 /data/nyh/Ascend/cann/set_env.sh，
# 没有自动生成 /data/nyh/Ascend/latest/set_env.sh。
# 若 /data/nyh/env.sh 仍 source latest，可以补一个兼容软链接。
cd /data/nyh/Ascend
ln -sfn cann latest
```

## 3. 使用方式
按照上述步骤安装完成后，每次建立新的shell session，可以设置以下环境变量：
```shell
source /usr/local/Ascend/driver/bin/setenv.bash
source /data/nyh/Ascend/latest/set_env.sh
export ASCEND_DEVICE_ID=7
```
我把上述代码保存到了`/data/nyh/env.sh`，可以通过命令`cd /data/nyh; source /data/nyh/env.sh`一键设置。
