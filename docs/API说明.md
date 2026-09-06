# API 说明

Base URL: `http://127.0.0.1:8000`

| 方法 | 路径 | 说明 | 参数 | 返回 |
| ---- | ---- | ---- | ---- | ---- |
| GET | `/` | 前端页面 | — | HTML |
| GET | `/api/model/info` | 模型信息 | — | 模型文件/设备/阈值/类别 |
| GET | `/api/dashboard` | 今日统计 | — | 检测数/违规数/正常数/违规率/类型分布 |
| GET | `/api/statistics` | 历史统计 | — | 每日违规/各类违规/严重程度分布 |
| GET | `/api/violations` | 违规记录查询 | `vtype`（类型）、`severity`、`date`（YYYY-MM-DD）、`limit`、`offset` | 记录列表（含截图 URL） |
| POST | `/api/detect/image` | 图片检测 | multipart: `file`（图片）、`lab_id` | 人员状态/违规事件/标注图 URL/耗时 |
| POST | `/api/detect/video` | 视频检测 | multipart: `file`（MP4）、`lab_id`、`stride`（默认2） | 帧数/FPS/违规事件/标注视频 URL |
| GET | `/api/settings` | 读取实验室/区域/规则 | — | 配置树 |
| POST | `/api/settings/lab` | 新建实验室 | form: `name`、`description` | ok |
| POST | `/api/settings/area` | 新建区域 | form: `lab_id`、`name`、`x1,y1,x2,y2`、`required_ppe`（逗号分隔） | ok |
| DELETE | `/api/settings/area/{id}` | 删除区域 | 路径参数 | ok |

静态资源：
- `/static/*` 前端资源
- `/media/screenshots/*` 违规截图
- `/media/annotated/*` 图片模式标注结果
- `/media/videos/*` 视频模式标注结果

### 示例

```bash
# 图片检测
curl -X POST http://127.0.0.1:8000/api/detect/image \
  -F "file=@data/test/t05_pcr_scientist.jpg" -F "lab_id=1"

# 视频检测（每3帧推理一次）
curl -X POST http://127.0.0.1:8000/api/detect/video \
  -F "file=@data/test/demo_lab_video.mp4" -F "lab_id=1" -F "stride=3"

# 查询"未佩戴口罩"违规
curl "http://127.0.0.1:8000/api/violations?vtype=未佩戴口罩"
```

交互式文档：启动后访问 `http://127.0.0.1:8000/docs`（FastAPI 自带 Swagger）。
