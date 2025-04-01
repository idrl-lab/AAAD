import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizer_adv.PC_DARTS.operations import *
from torch.autograd import Variable
from optimizer_adv.PC_DARTS.genotypes import PRIMITIVES
from optimizer_adv.PC_DARTS.genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:#PRIMITIVES中就是8个操作
      op = OPS[primitive](C, stride, False)#OPS中存储了各种操作的函数
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))#给池化操作后面加一个batchnormalization
      self._ops.append(op)#把这些op都放在预先定义好的modulelist里

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))
  # op(x)就是对输入x做一个相应的操作 w1*op1(x)+w2*op2(x)+...+w8*op8(x)
  # 也就是对输入x做8个操作并乘以相应的权重，把结果加起来


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__() #继承父类
    self.reduction = reduction
    #input nodes的结构固定不变，不参与搜索
    #决定第一个input nodes的结构，取决于前一个cell是否是reduction
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
      # 第一个input_nodes是cell k-2的输出，
      # cell k-2的输出通道数为C_prev_prev，所以这里操作的输入通道数为C_prev_prev
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    # 第二个input nodes的结构
    # 第二个input_nodes是cell k-1的输出
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    # 每个cell中有4个节点的连接状态待确定
    self._steps = steps
    self._multiplier = multiplier

    # 构建operation的modulelist
    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    # 遍历4个intermediate nodes构建混合操作
    for i in range(self._steps):
      # 遍历当前结点i的所有前驱节点
      for j in range(2+i):
        # 对第i个节点来说，他有j个前驱节点（每个节点的input都由前两个cell的输出和当前cell的前面的节点组成）
        stride = 2 if reduction and j < 2 else 1
        # op是构建两个节点之间的混合
        op = MixedOp(C, stride)
        # 所有边的混合操作添加到ops，list的len为2+3+4+5=14[[],[],...,[]]
        self._ops.append(op)

  # cell中的计算过程，前向传播时自动调用
  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]#当前节点的前驱节点
    offset = 0
    # 遍历每个intermediate nodes，得到每个节点的output
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
      #把当前节点i的output作为下一个节点的输入
      # states中为[s0,s1,b1,b2,b3,b4] b1,b2,b3,b4分别是四个intermediate output的输出
    # 对intermediate的output进行concat作为当前cell的输出
    # dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍
    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

