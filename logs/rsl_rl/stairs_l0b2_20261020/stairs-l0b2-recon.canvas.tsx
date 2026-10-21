import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const STEPS = [
  {
    n: "1",
    title: "只用当前这一帧",
    body: "观测里有 4 张 16×224 的距离图，时刻是 t−1.5 s、t−1.0 s、t−0.5 s、t。投格只切最后 3584 个数，也就是当前帧。前三帧不参与这次重建。",
  },
  {
    n: "2",
    title: "丢掉跨在棱上的光束",
    body: "沿俯仰或方位角看相邻光束。两边都是距离跳变、中间只有 1 到 2 条光束的那段被清掉，避免一条光束同时打到两个面。",
  },
  {
    n: "3",
    title: "射线落进 5 cm 格子",
    body: "方向乘上距离，变到世界坐标，再转到机器人当前朝向的机体系。高度图是 17×33，格子边长 5 cm。",
  },
  {
    n: "4",
    title: "格子里只留最高点往下 2 cm",
    body: "同一个格子里先找最高的回波，只把比它低不超过 2 cm 的点拿来平均。再低的当成另一层表面，不混进来。L0 一级大约 2 cm，所以低的那一层不会和踏面平均成中间高度。",
  },
  {
    n: "5",
    title: "近 1.4 秒取中值",
    body: "每隔约 0.2 秒的若干帧各自投一次，每个格子取这些高度的中值。只在某一帧出现的尖刺会被丢掉。",
  },
  {
    n: "6",
    title: "空格子才允许网络改",
    body: "没打中的格子先抄最近一个有回波格子的高度。网络可以在这个值上再加最多 8 cm。有回波的格子锁在第 5 步的结果上，delta、refine、edge_clean 都不改它。",
  },
  {
    n: "7",
    title: "整张图做 3×3 中值",
    body: "refined 最后过一次 3×3 中值。单个格子的颗粒被邻域高度替掉。真台阶比这个窗口宽，边缘留着。",
  },
];

export default function StairsL0b2Recon() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>楼梯高度重建 · l0b2</H1>
        <Text tone="secondary">
          2026-10-20 这一版。有回波的格子等于几何投格，网络只补空洞。副本在
          archive/stairs_l0b2_20261020。
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="16×224" label="当前帧，投格只用这一张" />
        <Stat value="5 cm" label="格子边长，输出 17×33" />
        <Stat value="2 cm" label="格内平均窗口，最高点往下" />
        <Stat value="±8 cm" label="只有空格子能动这么多" />
      </Grid>

      <H2>从距离到高度</H2>
      <Stack gap={12}>
        {STEPS.map((step) => (
          <Row key={step.n} gap={12} align="start">
            <Pill active size="sm">
              {step.n}
            </Pill>
            <Stack gap={4}>
              <H3>{step.title}</H3>
              <Text tone="secondary">{step.body}</Text>
            </Stack>
          </Row>
        ))}
      </Stack>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>图上两列分别是什么</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                rough：有回波处是投格中值，空洞处是邻近高度再加最多 8 cm。
              </Text>
              <Text>
                refined：空洞再经过 refine 和 edge_clean，然后整张图做 3×3 中值。有回波的格子和 rough 相同，中值滤波之后可能被邻居带过去一格。
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>损失为什么这样配</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                高度误差按整图高度跨度的平方根加权。L9 仍然比 L0 重，但不再是大约十倍。
              </Text>
              <Text>
                对齐台阶的那一项不乘这个权重。L0 一级只有大约 2 cm，只占一两对格子，如果再被跨度缩小，优化器会把它抹平。
              </Text>
              <Text tone="secondary">
                日志里的 L1 是不加权的绝对误差，单位是米。它不表示台阶还在不在。
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>最后一帧的 L1</H2>
        <Text tone="secondary" size="small">
          stairs_l0b2 · step 39999 · 每个数是该地形 64 个环境的平均绝对误差，单位米。图在
          docs/wtr_assets/yu_lidar_recon_stairs_l0b2/lidar_recon_step_39999.png。
        </Text>
        <Table
          headers={["地形", "高度差", "rough", "refined"]}
          columnAlign={["left", "right", "right", "right"]}
          rows={[
            ["平地", "0", "0.0077", "0.0035"],
            ["上 L0", "0.092", "0.0077", "0.0043"],
            ["下 L0", "0.046", "0.0070", "0.0032"],
            ["下 L4", "0.275", "0.0118", "0.0137"],
            ["上 L9", "0.683", "0.0239", "0.0121"],
            ["下 L9", "0.248", "0.0148", "0.0101"],
          ]}
          rowTone={[undefined, undefined, "warning", "warning", undefined, undefined]}
          striped
        />
      </Stack>

      <Callout tone="warning" title="下 L0 的数字小，台阶不一定还在">
        下 L0 这一帧 refined 是 0.0032，上 L0 是 0.0043。雷达朝下看时，近处的棱比下一级踏面更高，格子里的最高点是棱，低大约 2 cm 的踏面会被窗口丢掉。打不中的地方再抄棱的高度。有回波的格子锁死之后，网络补不回这 2 cm。误差摊在整张图上只有几毫米，色带却没了。
      </Callout>

      <Divider />
      <Text tone="tertiary" size="small">
        权重 lidar_reconstructor_phase1.pt。训练 40000 步，64 个环境，没有加载旧权重。
      </Text>
    </Stack>
  );
}
