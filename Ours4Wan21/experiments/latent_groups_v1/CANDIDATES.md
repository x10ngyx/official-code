# 十组候选 latent 特征：训练与在线推理共用定义

本轮把离线探针作为筛选线索，最终用真实 IQL 训练和 VBench 视频质量判断。选择 **10 个提取角度 + scalar5/SEA7 两个对照**，不把同一组的不同维度、cache/previous 拆分、联合384等重复拼接另算候选。不同角度可能相关，不声称统计独立；每个实验只添加下表一组。

## 统一输入时序与符号

- `x=x_t`：动作前当前候选输入 latent；`p=x_{t-1}`、`pp=x_{t-2}`：此前输入；`c=x_c`：最近一次实际 recompute 的输入。
- `v=(x-p)/(sigma_{t-1}-sigma_t)`：最近已完成的 latent 更新；`u=(p-pp)/(sigma_{t-2}-sigma_{t-1})`：再前一次更新。
- `w=(x_{c+1}-x_c)/(sigma_c-sigma_{c+1})`：缓存刷新步带来的更新。**`c+1<=t`，仅在它已经发生后使用；不是当前 reuse 后的未来输入。**
- `d=x-c`，`a=v-u`，`i=v-w`，分别为缓存漂移、更新加速度、相对缓存更新的创新。
- `R(z)`：每 channel 的时空 RMS，至少 `1e-6`；`cos(z,w)`：每 channel 的时空余弦；`S(z)=sign(z)log(1+abs(z))`，正数比率用 `log1p`。
- 所有 latent 特征先 `FP16→FP32`：本机既有存档为 FP16，在线也只对特征输入作同样量化，**不改变 DiT/scheduler 的输入和计算**。
- 50 步中每一步先 `observe` 后 `commit(实际动作)`，包括预算强制步；仅0清除历史和缓存（Wan2.1无第32步stage边界），未建立缓存时整组特征为零。49 的 latent 特征仍按当前输入计算，但 SEA 两个 distance 为历史规范的零哨兵；49 动作强制 recompute。
- 不读取 baseline、误差标签、最终 PSNR、未来动作、prompt ID。历史 baseline 误差及完整误差向量完全排除。

## 候选及精确提取方式

|编号 / CLI 名称|Latent维度|网络输入维度|角度与提取|
|---|---:|---:|---|
|1 `dynamics_raw_sea128`|128|135|raw/SEA 两域，每 channel 四项：`log1p(R(v)/R(c))`、`cos(v,u)`、`log1p(R(v-u)/R(u))`、`S(mean(v-u)/R(c))`；16×4×2。|
|2 `cache_update192`|192|199|最近更新与缓存刷新更新的幅度差、方向差、与漂移/加速度的对齐；16 channel×12 项，详见下方。保留 v9 最有希望候选的公式。|
|3 `local_drift1024`|1024|1031|对 `d` 取 **2×4×4** 时空块的有符号均值和 RMS，各除以该 channel 的 `R(c)` 后分别 S/log1p；16×32×2。比旧2×2×2块保留更多位置。|
|4 `spatial_gradient96`|96|103|沿视频时间、高、宽三轴差分，分别取 `log1p(R(∇x−∇c)/R(∇c))` 和 `cos(∇x,∇c)`；16×3×2。衡量结构/边缘变化，不是相邻 denoise 更新。|
|5 `channel_geometry240`|240|247|channel 时空中心化、除以各自标准差，计算16×16相关矩阵；取当前矩阵120个非对角上三角元素，以及当前−缓存的120个变化量。|
|6 `distribution256`|256|263|每 channel 的均值、标准差、5%/50%/95%分位数、偏度、峰度、`abs(zscore)>2`比例共8项；给当前值和相对缓存的变化，16×8×2。|
|7 `spectral_drift512`|512|519|完整 T/H/W FFT，16 channel×32频带：`log1p(sqrt(sum_band(abs(Fx−Fc)^2)/sum_band(abs(Fc)^2)))`。代表频带相对残差。|
|8 `spectral_phase512`|512|519|同32频带，去掉每个傅里叶系数幅值，仅平均 `1−cos(angle(Fx)−angle(Fc))`；忽略幅度≤eps的位置，无有效系数的频带取0。专门测试相位/位置变化。|
|9 `spectral_shape576`|576|583|每 channel 的32频带能量归一化成概率：取当前−缓存的512个概率变化；再加每channel当前熵、熵变化、top4频带能量占比、占比变化，共64项。总576。|
|10 `spectral_dynamics1024`|1024|1031|分别对 `a=v-u` 与 `i=v-w` 做FFT，取32频带相对 RMS：前者以 `F(u)` 能量归一，后者以 `F(w)` 能量归一；两套512拼为一组，代表频域加速度与更新失配。|
|对照 `sea7`|0|7|仅原SEA7，网络不接受latent输入。|
|对照 `scalar5`|0|5|五标量，网络不接受SEA距离或latent输入。|

### SEA 动态与曲率口径

SEA 组对 `x,p,pp,c` **都使用当前 sigma 的同一个滤波器**再差分，避免把滤波器随 sigma 变化本身混入“动态”。滤波器为已有 SEA 可分离3D FFT公式：每轴 `P(f)=1/(abs(f)^3+1e-16)`，`G(f)=(1−sigma)P / ((1−sigma)^2 P+sigma^2+1e-16)`，三轴相乘后除以全频均值。

raw/SEA 两域放在同一个组，便于控制总实验数量；该组如有效，不能直接归因于其中某一个域。这里的“曲率”指相对更新变化的代理，不是严格微分几何曲率。

### cache_update192 的12项/channel

按顺序：

1. `log1p(R(v)/R(c))`
2. `log1p(R(w)/R(c))`
3. `log1p(R(i)/R(w))`
4. `cos(v,w)`
5. `cos(i,d)`
6. `cos(w,d)`
7. `cos(a,w)`
8. `log1p(R(a)/R(w))`
9. `S(mean(i)/R(c))`
10. `S(mean(v*w)/R(c)^2)`
11. `log1p(R(d)/R(c))`
12. `cos(i,x)`

### 值域与频带细节

值域统计中的 mean/quantile 以缓存 channel RMS 归一后作 S；std 归一后 log1p；偏度作 S；峰度为非超额四阶标准矩并作 log1p；尾部比例不变换。`当前−缓存` 是上述变换之后的差，不另给baseline信息。

32频带=4个视频时间频率桶×8个空间径向桶。时间轴绝对频率索引按边界0/2/4分桶，空间 `f_h²+f_w²` 按 `j²/128, j=1..7` 分桶，与此前 probe 的32频带定义一致。空频带能量为0，分母有eps保护。频谱形状的熵除以 `log(32)`，top4占比在channel内计算。


## Wan2.1 接口

实验组state-mode为 `sea7_` 加上表中的CLI名称；输入顺序是latent feature在前、原SEA7在后。参数沿用父README的本机400epoch训练设置。权重记录完整特征版本、维度、量化、时序合同；不兼容Wan2.2 checkpoint。SEA公式由同级SeaCache4Wan21核对，原始来源及修改见项目latent_feature_lock.json。
