# 🎦 Xiaomi Camera Streamer


## Install / 安装

### 🐳 Docker compose
```shell
mkdir /opt/micam
cd /opt/micam
wget https://raw.githubusercontent.com/miiot/micam/refs/heads/main/docker-compose.yml
docker compose up -d
```

> 此命令会通过docker部署Miloco、Go2rtc及RTSP转发服务。如果需要添加多个摄像头，需要编辑`docker-compose.yml`运行多个micam服务。
>
> 部署的Miloco为基础版，不带AI引擎，无GPU算力要求，大部分机器都能运行。本项目基于官方镜像修改，添加了arm64支持，并默认获取高清流和音频流。


## 💻 Usage / 使用

### [Miloco](https://github.com/XiaoMi/xiaomi-miloco)

> 🏠 你也可以选择通过[HAOS加载项](https://gitee.com/hasscc/addons)来部署Miloco，[一键添加](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgitee.com%2Fhasscc%2Faddons)加载项仓库。

1. Open Miloco WebUI / 打开Miloco网页: `https://192.168.1.xx:8000`
   > 🔐 Miloco使用了自签证书，请通过 **https** 访问，并忽略证书错误。
2. Set miloco password / 设置Miloco密码
3. Bind your Xiaomi account / 绑定小米账号
4. Camera offline ? [[Xiaomi Miloco Q&A]](https://github.com/XiaoMi/xiaomi-miloco/issues/56)


### [Go2rtc](https://github.com/AlexxIT/go2rtc)

> 🏠 你也可以选择通过[HAOS加载项](https://github.com/AlexxIT/hassio-addons)来部署Go2rtc

1. Open Go2rtc WebUI / 访问Go2rtc网页: `http://192.168.1.xx:1984/config.html`
2. Config empty streams / 配置空视频流:
   ```yaml
   streams:
     your_stream1:
     your_stream2:
   ```
3. Save & Restart / 保存并重启


### [Micam](https://zread.ai/miiot/micam)

1. Set environment variables / 设置环境变量:
   ```shell
   cat << EOF > .env
   MILOCO_PASSWORD=your_miloco_password_md5
   CAMERA_ID=1234567890 # your camera did
   RTSP_URL=rtsp://192.168.1.xx:8554/your_stream1
   EOF
   ```
2. Restart micam / 重启转发服务: `docker compose restart micam1`


## ⚙️ Configuration / 配置

### Environments / 环境变量

> [!Note]
> 建议所有的环境变量配置在`.env`文件中，并使用`docker compose up -d`命令使其生效，不建议直接修改`docker-compose.yml`中的环境变量。

1. Micam:
   - `MILOCO_BASE_URL`: Miloco Base URL, Default: `https://miloco:8000`
     > 如果通过[HAOS加载项](https://gitee.com/hasscc/addons)部署，则应配置为`https://homeassistant.local:28800`
   - `MILOCO_PASSWORD`: Miloco WebUI Password (md5/lower), Required
   - `CAMERA_ID`: Camera DID, Required
     > 可在Miloco网页中通过F12开发者工具的网络请求日志查看
   - `RTSP_URL`: RTSP URL, Required
     > 转推RTSP流地址，如: `rtsp://192.168.1.xx:8554/your_stream1`，8554为Go2rtc提供的RTSP服务
   - `VIDEO_CODEC`: Video Codec of the camera, `hevc`(default) or `h264`
   - `STREAM_CHANNEL`: Stream Channel of the camera, Default: `0`

2. Miloco:
   - `MILOCO_PORT`: Miloco listen port, Default: `8000`
     > 如果与其他服务端口冲突，请修改此端口，并修改`MILOCO_BASE_URL`
   - `MILOCO_HOST`: Miloco listen host, Default: `0.0.0.0`, Don't change
   - `MILOCO_LOG_LEVEL`: Miloco log level, Default: `warning`

3. Watchdog:
   - `WATCH_STREAMS`: Comma separated go2rtc stream names to watch, e.g. `stream1,stream2`, Required
     > 视频停止3分钟后让Miloco刷新摄像头，6分钟后重启Miloco一次（30分钟内最多一次）；重启后仍无视频则需给摄像头断电重启
   - `GO2RTC_URL`: Go2rtc API URL, Default: `http://go2rtc:1984`


## 🧩 Integrations / 集成
- [Home Assistant: Generic Camera](https://www.home-assistant.io/integrations/generic)
- [Frigate NVR](https://github.com/blakeblackshear/frigate): [HAOS Add-on](https://github.com/blakeblackshear/frigate-hass-addons)
- [Scrypted](https://github.com/koush/scrypted): [HAOS Add-on](https://github.com/koush/scrypted/wiki/Installation:-Home-Assistant-OS)


## 🔗 Links / 相关链接
- [详细部署文档 & AI问答](https://zread.ai/miiot/micam)
- [Xiaomi Miloco](https://github.com/XiaoMi/xiaomi-miloco)
- [AlexxIT Go2rtc](https://github.com/AlexxIT/go2rtc)
