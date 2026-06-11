#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> ()>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map9 = affine_map<(d0, d1, d2) -> (d2)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @kernel(%arg0: tensor<1x8xi64>) -> tensor<1x8x151936xf32> {
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %c151936 = arith.constant 151936 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense<-3.40282347E+38> : tensor<f32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1x8x8xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_896_torch.float32_72> : tensor<896xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_23> : tensor<896x4864xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_47> : tensor<4864x896xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_46> : tensor<4864x896xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_896_torch.float32_71> : tensor<896xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_47> : tensor<896x896xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_128_torch.float32_47> : tensor<128xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_47> : tensor<128x896xf32>
    %cst_12 = arith.constant dense_resource<torch_tensor_128_torch.float32_46> : tensor<128xf32>
    %cst_13 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_46> : tensor<128x896xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_896_torch.float32_70> : tensor<896xf32>
    %cst_15 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_46> : tensor<896x896xf32>
    %cst_16 = arith.constant dense_resource<torch_tensor_896_torch.float32_69> : tensor<896xf32>
    %cst_17 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_22> : tensor<896x4864xf32>
    %cst_18 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_45> : tensor<4864x896xf32>
    %cst_19 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_44> : tensor<4864x896xf32>
    %cst_20 = arith.constant dense_resource<torch_tensor_896_torch.float32_68> : tensor<896xf32>
    %cst_21 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_45> : tensor<896x896xf32>
    %cst_22 = arith.constant dense_resource<torch_tensor_128_torch.float32_45> : tensor<128xf32>
    %cst_23 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_45> : tensor<128x896xf32>
    %cst_24 = arith.constant dense_resource<torch_tensor_128_torch.float32_44> : tensor<128xf32>
    %cst_25 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_44> : tensor<128x896xf32>
    %cst_26 = arith.constant dense_resource<torch_tensor_896_torch.float32_67> : tensor<896xf32>
    %cst_27 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_44> : tensor<896x896xf32>
    %cst_28 = arith.constant dense_resource<torch_tensor_896_torch.float32_66> : tensor<896xf32>
    %cst_29 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_21> : tensor<896x4864xf32>
    %cst_30 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_43> : tensor<4864x896xf32>
    %cst_31 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_42> : tensor<4864x896xf32>
    %cst_32 = arith.constant dense_resource<torch_tensor_896_torch.float32_65> : tensor<896xf32>
    %cst_33 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_43> : tensor<896x896xf32>
    %cst_34 = arith.constant dense_resource<torch_tensor_128_torch.float32_43> : tensor<128xf32>
    %cst_35 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_43> : tensor<128x896xf32>
    %cst_36 = arith.constant dense_resource<torch_tensor_128_torch.float32_42> : tensor<128xf32>
    %cst_37 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_42> : tensor<128x896xf32>
    %cst_38 = arith.constant dense_resource<torch_tensor_896_torch.float32_64> : tensor<896xf32>
    %cst_39 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_42> : tensor<896x896xf32>
    %cst_40 = arith.constant dense_resource<torch_tensor_896_torch.float32_63> : tensor<896xf32>
    %cst_41 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_20> : tensor<896x4864xf32>
    %cst_42 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_41> : tensor<4864x896xf32>
    %cst_43 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_40> : tensor<4864x896xf32>
    %cst_44 = arith.constant dense_resource<torch_tensor_896_torch.float32_62> : tensor<896xf32>
    %cst_45 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_41> : tensor<896x896xf32>
    %cst_46 = arith.constant dense_resource<torch_tensor_128_torch.float32_41> : tensor<128xf32>
    %cst_47 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_41> : tensor<128x896xf32>
    %cst_48 = arith.constant dense_resource<torch_tensor_128_torch.float32_40> : tensor<128xf32>
    %cst_49 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_40> : tensor<128x896xf32>
    %cst_50 = arith.constant dense_resource<torch_tensor_896_torch.float32_61> : tensor<896xf32>
    %cst_51 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_40> : tensor<896x896xf32>
    %cst_52 = arith.constant dense_resource<torch_tensor_896_torch.float32_60> : tensor<896xf32>
    %cst_53 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_19> : tensor<896x4864xf32>
    %cst_54 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_39> : tensor<4864x896xf32>
    %cst_55 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_38> : tensor<4864x896xf32>
    %cst_56 = arith.constant dense_resource<torch_tensor_896_torch.float32_59> : tensor<896xf32>
    %cst_57 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_39> : tensor<896x896xf32>
    %cst_58 = arith.constant dense_resource<torch_tensor_128_torch.float32_39> : tensor<128xf32>
    %cst_59 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_39> : tensor<128x896xf32>
    %cst_60 = arith.constant dense_resource<torch_tensor_128_torch.float32_38> : tensor<128xf32>
    %cst_61 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_38> : tensor<128x896xf32>
    %cst_62 = arith.constant dense_resource<torch_tensor_896_torch.float32_58> : tensor<896xf32>
    %cst_63 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_38> : tensor<896x896xf32>
    %cst_64 = arith.constant dense_resource<torch_tensor_896_torch.float32_57> : tensor<896xf32>
    %cst_65 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_18> : tensor<896x4864xf32>
    %cst_66 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_37> : tensor<4864x896xf32>
    %cst_67 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_36> : tensor<4864x896xf32>
    %cst_68 = arith.constant dense_resource<torch_tensor_896_torch.float32_56> : tensor<896xf32>
    %cst_69 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_37> : tensor<896x896xf32>
    %cst_70 = arith.constant dense_resource<torch_tensor_128_torch.float32_37> : tensor<128xf32>
    %cst_71 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_37> : tensor<128x896xf32>
    %cst_72 = arith.constant dense_resource<torch_tensor_128_torch.float32_36> : tensor<128xf32>
    %cst_73 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_36> : tensor<128x896xf32>
    %cst_74 = arith.constant dense_resource<torch_tensor_896_torch.float32_55> : tensor<896xf32>
    %cst_75 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_36> : tensor<896x896xf32>
    %cst_76 = arith.constant dense_resource<torch_tensor_896_torch.float32_54> : tensor<896xf32>
    %cst_77 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_17> : tensor<896x4864xf32>
    %cst_78 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_35> : tensor<4864x896xf32>
    %cst_79 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_34> : tensor<4864x896xf32>
    %cst_80 = arith.constant dense_resource<torch_tensor_896_torch.float32_53> : tensor<896xf32>
    %cst_81 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_35> : tensor<896x896xf32>
    %cst_82 = arith.constant dense_resource<torch_tensor_128_torch.float32_35> : tensor<128xf32>
    %cst_83 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_35> : tensor<128x896xf32>
    %cst_84 = arith.constant dense_resource<torch_tensor_128_torch.float32_34> : tensor<128xf32>
    %cst_85 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_34> : tensor<128x896xf32>
    %cst_86 = arith.constant dense_resource<torch_tensor_896_torch.float32_52> : tensor<896xf32>
    %cst_87 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_34> : tensor<896x896xf32>
    %cst_88 = arith.constant dense_resource<torch_tensor_896_torch.float32_51> : tensor<896xf32>
    %cst_89 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_16> : tensor<896x4864xf32>
    %cst_90 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_33> : tensor<4864x896xf32>
    %cst_91 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_32> : tensor<4864x896xf32>
    %cst_92 = arith.constant dense_resource<torch_tensor_896_torch.float32_50> : tensor<896xf32>
    %cst_93 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_33> : tensor<896x896xf32>
    %cst_94 = arith.constant dense_resource<torch_tensor_128_torch.float32_33> : tensor<128xf32>
    %cst_95 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_33> : tensor<128x896xf32>
    %cst_96 = arith.constant dense_resource<torch_tensor_128_torch.float32_32> : tensor<128xf32>
    %cst_97 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_32> : tensor<128x896xf32>
    %cst_98 = arith.constant dense_resource<torch_tensor_896_torch.float32_49> : tensor<896xf32>
    %cst_99 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_32> : tensor<896x896xf32>
    %cst_100 = arith.constant dense_resource<torch_tensor_896_torch.float32_48> : tensor<896xf32>
    %cst_101 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_15> : tensor<896x4864xf32>
    %cst_102 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_31> : tensor<4864x896xf32>
    %cst_103 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_30> : tensor<4864x896xf32>
    %cst_104 = arith.constant dense_resource<torch_tensor_896_torch.float32_47> : tensor<896xf32>
    %cst_105 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_31> : tensor<896x896xf32>
    %cst_106 = arith.constant dense_resource<torch_tensor_128_torch.float32_31> : tensor<128xf32>
    %cst_107 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_31> : tensor<128x896xf32>
    %cst_108 = arith.constant dense_resource<torch_tensor_128_torch.float32_30> : tensor<128xf32>
    %cst_109 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_30> : tensor<128x896xf32>
    %cst_110 = arith.constant dense_resource<torch_tensor_896_torch.float32_46> : tensor<896xf32>
    %cst_111 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_30> : tensor<896x896xf32>
    %cst_112 = arith.constant dense_resource<torch_tensor_896_torch.float32_45> : tensor<896xf32>
    %cst_113 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_14> : tensor<896x4864xf32>
    %cst_114 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_29> : tensor<4864x896xf32>
    %cst_115 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_28> : tensor<4864x896xf32>
    %cst_116 = arith.constant dense_resource<torch_tensor_896_torch.float32_44> : tensor<896xf32>
    %cst_117 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_29> : tensor<896x896xf32>
    %cst_118 = arith.constant dense_resource<torch_tensor_128_torch.float32_29> : tensor<128xf32>
    %cst_119 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_29> : tensor<128x896xf32>
    %cst_120 = arith.constant dense_resource<torch_tensor_128_torch.float32_28> : tensor<128xf32>
    %cst_121 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_28> : tensor<128x896xf32>
    %cst_122 = arith.constant dense_resource<torch_tensor_896_torch.float32_43> : tensor<896xf32>
    %cst_123 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_28> : tensor<896x896xf32>
    %cst_124 = arith.constant dense_resource<torch_tensor_896_torch.float32_42> : tensor<896xf32>
    %cst_125 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_13> : tensor<896x4864xf32>
    %cst_126 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_27> : tensor<4864x896xf32>
    %cst_127 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_26> : tensor<4864x896xf32>
    %cst_128 = arith.constant dense_resource<torch_tensor_896_torch.float32_41> : tensor<896xf32>
    %cst_129 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_27> : tensor<896x896xf32>
    %cst_130 = arith.constant dense_resource<torch_tensor_128_torch.float32_27> : tensor<128xf32>
    %cst_131 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_27> : tensor<128x896xf32>
    %cst_132 = arith.constant dense_resource<torch_tensor_128_torch.float32_26> : tensor<128xf32>
    %cst_133 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_26> : tensor<128x896xf32>
    %cst_134 = arith.constant dense_resource<torch_tensor_896_torch.float32_40> : tensor<896xf32>
    %cst_135 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_26> : tensor<896x896xf32>
    %cst_136 = arith.constant dense_resource<torch_tensor_896_torch.float32_39> : tensor<896xf32>
    %cst_137 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_12> : tensor<896x4864xf32>
    %cst_138 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_25> : tensor<4864x896xf32>
    %cst_139 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_24> : tensor<4864x896xf32>
    %cst_140 = arith.constant dense_resource<torch_tensor_896_torch.float32_38> : tensor<896xf32>
    %cst_141 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_25> : tensor<896x896xf32>
    %cst_142 = arith.constant dense_resource<torch_tensor_128_torch.float32_25> : tensor<128xf32>
    %cst_143 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_25> : tensor<128x896xf32>
    %cst_144 = arith.constant dense_resource<torch_tensor_128_torch.float32_24> : tensor<128xf32>
    %cst_145 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_24> : tensor<128x896xf32>
    %cst_146 = arith.constant dense_resource<torch_tensor_896_torch.float32_37> : tensor<896xf32>
    %cst_147 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_24> : tensor<896x896xf32>
    %cst_148 = arith.constant dense_resource<torch_tensor_896_torch.float32_36> : tensor<896xf32>
    %cst_149 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_11> : tensor<896x4864xf32>
    %cst_150 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_23> : tensor<4864x896xf32>
    %cst_151 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_22> : tensor<4864x896xf32>
    %cst_152 = arith.constant dense_resource<torch_tensor_896_torch.float32_35> : tensor<896xf32>
    %cst_153 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_23> : tensor<896x896xf32>
    %cst_154 = arith.constant dense_resource<torch_tensor_128_torch.float32_23> : tensor<128xf32>
    %cst_155 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_23> : tensor<128x896xf32>
    %cst_156 = arith.constant dense_resource<torch_tensor_128_torch.float32_22> : tensor<128xf32>
    %cst_157 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_22> : tensor<128x896xf32>
    %cst_158 = arith.constant dense_resource<torch_tensor_896_torch.float32_34> : tensor<896xf32>
    %cst_159 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_22> : tensor<896x896xf32>
    %cst_160 = arith.constant dense_resource<torch_tensor_896_torch.float32_33> : tensor<896xf32>
    %cst_161 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_10> : tensor<896x4864xf32>
    %cst_162 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_21> : tensor<4864x896xf32>
    %cst_163 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_20> : tensor<4864x896xf32>
    %cst_164 = arith.constant dense_resource<torch_tensor_896_torch.float32_32> : tensor<896xf32>
    %cst_165 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_21> : tensor<896x896xf32>
    %cst_166 = arith.constant dense_resource<torch_tensor_128_torch.float32_21> : tensor<128xf32>
    %cst_167 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_21> : tensor<128x896xf32>
    %cst_168 = arith.constant dense_resource<torch_tensor_128_torch.float32_20> : tensor<128xf32>
    %cst_169 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_20> : tensor<128x896xf32>
    %cst_170 = arith.constant dense_resource<torch_tensor_896_torch.float32_31> : tensor<896xf32>
    %cst_171 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_20> : tensor<896x896xf32>
    %cst_172 = arith.constant dense_resource<torch_tensor_896_torch.float32_30> : tensor<896xf32>
    %cst_173 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_9> : tensor<896x4864xf32>
    %cst_174 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_19> : tensor<4864x896xf32>
    %cst_175 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_18> : tensor<4864x896xf32>
    %cst_176 = arith.constant dense_resource<torch_tensor_896_torch.float32_29> : tensor<896xf32>
    %cst_177 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_19> : tensor<896x896xf32>
    %cst_178 = arith.constant dense_resource<torch_tensor_128_torch.float32_19> : tensor<128xf32>
    %cst_179 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_19> : tensor<128x896xf32>
    %cst_180 = arith.constant dense_resource<torch_tensor_128_torch.float32_18> : tensor<128xf32>
    %cst_181 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_18> : tensor<128x896xf32>
    %cst_182 = arith.constant dense_resource<torch_tensor_896_torch.float32_28> : tensor<896xf32>
    %cst_183 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_18> : tensor<896x896xf32>
    %cst_184 = arith.constant dense_resource<torch_tensor_896_torch.float32_27> : tensor<896xf32>
    %cst_185 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_8> : tensor<896x4864xf32>
    %cst_186 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_17> : tensor<4864x896xf32>
    %cst_187 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_16> : tensor<4864x896xf32>
    %cst_188 = arith.constant dense_resource<torch_tensor_896_torch.float32_26> : tensor<896xf32>
    %cst_189 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_17> : tensor<896x896xf32>
    %cst_190 = arith.constant dense_resource<torch_tensor_128_torch.float32_17> : tensor<128xf32>
    %cst_191 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_17> : tensor<128x896xf32>
    %cst_192 = arith.constant dense_resource<torch_tensor_128_torch.float32_16> : tensor<128xf32>
    %cst_193 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_16> : tensor<128x896xf32>
    %cst_194 = arith.constant dense_resource<torch_tensor_896_torch.float32_25> : tensor<896xf32>
    %cst_195 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_16> : tensor<896x896xf32>
    %cst_196 = arith.constant dense_resource<torch_tensor_896_torch.float32_24> : tensor<896xf32>
    %cst_197 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_7> : tensor<896x4864xf32>
    %cst_198 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_15> : tensor<4864x896xf32>
    %cst_199 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_14> : tensor<4864x896xf32>
    %cst_200 = arith.constant dense_resource<torch_tensor_896_torch.float32_23> : tensor<896xf32>
    %cst_201 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_15> : tensor<896x896xf32>
    %cst_202 = arith.constant dense_resource<torch_tensor_128_torch.float32_15> : tensor<128xf32>
    %cst_203 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_15> : tensor<128x896xf32>
    %cst_204 = arith.constant dense_resource<torch_tensor_128_torch.float32_14> : tensor<128xf32>
    %cst_205 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_14> : tensor<128x896xf32>
    %cst_206 = arith.constant dense_resource<torch_tensor_896_torch.float32_22> : tensor<896xf32>
    %cst_207 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_14> : tensor<896x896xf32>
    %cst_208 = arith.constant dense_resource<torch_tensor_896_torch.float32_21> : tensor<896xf32>
    %cst_209 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_6> : tensor<896x4864xf32>
    %cst_210 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_13> : tensor<4864x896xf32>
    %cst_211 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_12> : tensor<4864x896xf32>
    %cst_212 = arith.constant dense_resource<torch_tensor_896_torch.float32_20> : tensor<896xf32>
    %cst_213 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_13> : tensor<896x896xf32>
    %cst_214 = arith.constant dense_resource<torch_tensor_128_torch.float32_13> : tensor<128xf32>
    %cst_215 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_13> : tensor<128x896xf32>
    %cst_216 = arith.constant dense_resource<torch_tensor_128_torch.float32_12> : tensor<128xf32>
    %cst_217 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_12> : tensor<128x896xf32>
    %cst_218 = arith.constant dense_resource<torch_tensor_896_torch.float32_19> : tensor<896xf32>
    %cst_219 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_12> : tensor<896x896xf32>
    %cst_220 = arith.constant dense_resource<torch_tensor_896_torch.float32_18> : tensor<896xf32>
    %cst_221 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_5> : tensor<896x4864xf32>
    %cst_222 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_11> : tensor<4864x896xf32>
    %cst_223 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_10> : tensor<4864x896xf32>
    %cst_224 = arith.constant dense_resource<torch_tensor_896_torch.float32_17> : tensor<896xf32>
    %cst_225 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_11> : tensor<896x896xf32>
    %cst_226 = arith.constant dense_resource<torch_tensor_128_torch.float32_11> : tensor<128xf32>
    %cst_227 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_11> : tensor<128x896xf32>
    %cst_228 = arith.constant dense_resource<torch_tensor_128_torch.float32_10> : tensor<128xf32>
    %cst_229 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_10> : tensor<128x896xf32>
    %cst_230 = arith.constant dense_resource<torch_tensor_896_torch.float32_16> : tensor<896xf32>
    %cst_231 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_10> : tensor<896x896xf32>
    %cst_232 = arith.constant dense_resource<torch_tensor_896_torch.float32_15> : tensor<896xf32>
    %cst_233 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_4> : tensor<896x4864xf32>
    %cst_234 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_9> : tensor<4864x896xf32>
    %cst_235 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_8> : tensor<4864x896xf32>
    %cst_236 = arith.constant dense_resource<torch_tensor_896_torch.float32_14> : tensor<896xf32>
    %cst_237 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_9> : tensor<896x896xf32>
    %cst_238 = arith.constant dense_resource<torch_tensor_128_torch.float32_9> : tensor<128xf32>
    %cst_239 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_9> : tensor<128x896xf32>
    %cst_240 = arith.constant dense_resource<torch_tensor_128_torch.float32_8> : tensor<128xf32>
    %cst_241 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_8> : tensor<128x896xf32>
    %cst_242 = arith.constant dense_resource<torch_tensor_896_torch.float32_13> : tensor<896xf32>
    %cst_243 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_8> : tensor<896x896xf32>
    %cst_244 = arith.constant dense_resource<torch_tensor_896_torch.float32_12> : tensor<896xf32>
    %cst_245 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_3> : tensor<896x4864xf32>
    %cst_246 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_7> : tensor<4864x896xf32>
    %cst_247 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_6> : tensor<4864x896xf32>
    %cst_248 = arith.constant dense_resource<torch_tensor_896_torch.float32_11> : tensor<896xf32>
    %cst_249 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_7> : tensor<896x896xf32>
    %cst_250 = arith.constant dense_resource<torch_tensor_128_torch.float32_7> : tensor<128xf32>
    %cst_251 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_7> : tensor<128x896xf32>
    %cst_252 = arith.constant dense_resource<torch_tensor_128_torch.float32_6> : tensor<128xf32>
    %cst_253 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_6> : tensor<128x896xf32>
    %cst_254 = arith.constant dense_resource<torch_tensor_896_torch.float32_10> : tensor<896xf32>
    %cst_255 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_6> : tensor<896x896xf32>
    %cst_256 = arith.constant dense_resource<torch_tensor_896_torch.float32_9> : tensor<896xf32>
    %cst_257 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_2> : tensor<896x4864xf32>
    %cst_258 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_5> : tensor<4864x896xf32>
    %cst_259 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_4> : tensor<4864x896xf32>
    %cst_260 = arith.constant dense_resource<torch_tensor_896_torch.float32_8> : tensor<896xf32>
    %cst_261 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_5> : tensor<896x896xf32>
    %cst_262 = arith.constant dense_resource<torch_tensor_128_torch.float32_5> : tensor<128xf32>
    %cst_263 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_5> : tensor<128x896xf32>
    %cst_264 = arith.constant dense_resource<torch_tensor_128_torch.float32_4> : tensor<128xf32>
    %cst_265 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_4> : tensor<128x896xf32>
    %cst_266 = arith.constant dense_resource<torch_tensor_896_torch.float32_7> : tensor<896xf32>
    %cst_267 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_4> : tensor<896x896xf32>
    %cst_268 = arith.constant dense_resource<torch_tensor_896_torch.float32_6> : tensor<896xf32>
    %cst_269 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32_1> : tensor<896x4864xf32>
    %cst_270 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_3> : tensor<4864x896xf32>
    %cst_271 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_2> : tensor<4864x896xf32>
    %cst_272 = arith.constant dense_resource<torch_tensor_896_torch.float32_5> : tensor<896xf32>
    %cst_273 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_3> : tensor<896x896xf32>
    %cst_274 = arith.constant dense_resource<torch_tensor_128_torch.float32_3> : tensor<128xf32>
    %cst_275 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_3> : tensor<128x896xf32>
    %cst_276 = arith.constant dense_resource<torch_tensor_128_torch.float32_2> : tensor<128xf32>
    %cst_277 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_2> : tensor<128x896xf32>
    %cst_278 = arith.constant dense_resource<torch_tensor_896_torch.float32_4> : tensor<896xf32>
    %cst_279 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_2> : tensor<896x896xf32>
    %cst_280 = arith.constant dense_resource<torch_tensor_896_torch.float32_3> : tensor<896xf32>
    %cst_281 = arith.constant dense_resource<torch_tensor_896_4864_torch.float32> : tensor<896x4864xf32>
    %cst_282 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32_1> : tensor<4864x896xf32>
    %cst_283 = arith.constant dense_resource<torch_tensor_4864_896_torch.float32> : tensor<4864x896xf32>
    %cst_284 = arith.constant dense_resource<torch_tensor_896_torch.float32_2> : tensor<896xf32>
    %cst_285 = arith.constant dense_resource<torch_tensor_896_896_torch.float32_1> : tensor<896x896xf32>
    %cst_286 = arith.constant dense_resource<torch_tensor_128_torch.float32_1> : tensor<128xf32>
    %cst_287 = arith.constant dense_resource<torch_tensor_128_896_torch.float32_1> : tensor<128x896xf32>
    %cst_288 = arith.constant dense_resource<torch_tensor_128_torch.float32> : tensor<128xf32>
    %cst_289 = arith.constant dense_resource<torch_tensor_128_896_torch.float32> : tensor<128x896xf32>
    %cst_290 = arith.constant dense_resource<torch_tensor_896_torch.float32_1> : tensor<896xf32>
    %cst_291 = arith.constant dense_resource<torch_tensor_896_896_torch.float32> : tensor<896x896xf32>
    %cst_292 = arith.constant dense_resource<torch_tensor_896_torch.float32> : tensor<896xf32>
    %cst_293 = arith.constant 9.9999999999999995E-7 : f64
    %cst_294 = arith.constant dense_resource<torch_tensor_32_torch.float32> : tensor<32xf32>
    %cst_295 = arith.constant dense_resource<torch_tensor_151936_896_torch.float32> : tensor<151936x896xf32>
    %cst_296 = arith.constant 8.960000e+02 : f32
    %cst_297 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<8xi64>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<8xi64>) {
    ^bb0(%out: i64):
      %1381 = linalg.index 0 : index
      %1382 = arith.index_cast %1381 : index to i64
      linalg.yield %1382 : i64
    } -> tensor<8xi64>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [1, 8] : tensor<8xi64> into tensor<1x8xi64>
    %2 = tensor.empty() : tensor<1x8x896xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x8xi64>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: i64, %out: f32):
      %1381 = arith.index_cast %in : i64 to index
      %1382 = linalg.index 2 : index
      %1383 = arith.cmpi slt, %1381, %c151936 : index
      cf.assert %1383, "index must be smaller than dim size"
      %1384 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %1384, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_295[%1381, %1382] : tensor<151936x896xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x8x896xf32>
    %expanded_298 = tensor.expand_shape %1 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi64> into tensor<1x1x1x8xi64>
    %expanded_299 = tensor.expand_shape %1 [[0, 1, 2, 3]] output_shape [1, 1, 8, 1] : tensor<8xi64> into tensor<1x1x8x1xi64>
    %4 = tensor.empty() : tensor<1x1x8x8xi1>
    %5 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_298, %expanded_299 : tensor<1x1x1x8xi64>, tensor<1x1x8x1xi64>) outs(%4 : tensor<1x1x8x8xi1>) {
    ^bb0(%in: i64, %in_1021: i64, %out: i1):
      %1381 = arith.cmpi sgt, %in, %in_1021 : i64
      linalg.yield %1381 : i1
    } -> tensor<1x1x8x8xi1>
    %6 = tensor.empty() : tensor<1x1x8x8xf32>
    %7 = linalg.generic {indexing_maps = [#map5, #map6, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst_2, %cst_3 : tensor<1x1x8x8xi1>, tensor<f32>, tensor<1x1x8x8xf32>) outs(%6 : tensor<1x1x8x8xf32>) {
    ^bb0(%in: i1, %in_1021: f32, %in_1022: f32, %out: f32):
      %1381 = arith.select %in, %in_1021, %in_1022 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x1x8x8xf32>
    %8 = tensor.empty() : tensor<1x8xf32>
    %9 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<1x8xi64>) outs(%8 : tensor<1x8xf32>) {
    ^bb0(%in: i64, %out: f32):
      %1381 = arith.sitofp %in : i64 to f32
      linalg.yield %1381 : f32
    } -> tensor<1x8xf32>
    %expanded_300 = tensor.expand_shape %9 [[0], [1, 2]] output_shape [1, 8, 1] : tensor<1x8xf32> into tensor<1x8x1xf32>
    %10 = tensor.empty() : tensor<1x8x32xf32>
    %11 = linalg.generic {indexing_maps = [#map8, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_300, %cst_294 : tensor<1x8x1xf32>, tensor<32xf32>) outs(%10 : tensor<1x8x32xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x32xf32>
    %concat = tensor.concat dim(2) %11, %11 : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x64xf32>
    %12 = tensor.empty() : tensor<1x8x64xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%concat : tensor<1x8x64xf32>) outs(%12 : tensor<1x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.cos %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x64xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%concat : tensor<1x8x64xf32>) outs(%12 : tensor<1x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.sin %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x64xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %16 = tensor.empty() : tensor<1x8x1xf32>
    %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %22 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %21 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %23 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_292, %22 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %24 = tensor.empty() : tensor<896x896xf32>
    %transposed = linalg.transpose ins(%cst_291 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %25 = tensor.empty() : tensor<1x896x896xf32>
    %26 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %27 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %28 = linalg.batch_matmul ins(%23, %26 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %29 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %cst_290 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_301 = tensor.expand_shape %29 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %30 = tensor.empty() : tensor<1x14x8x64xf32>
    %transposed_302 = linalg.transpose ins(%expanded_301 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %31 = tensor.empty() : tensor<896x128xf32>
    %transposed_303 = linalg.transpose ins(%cst_289 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %32 = tensor.empty() : tensor<1x896x128xf32>
    %33 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_303 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %34 = tensor.empty() : tensor<1x8x128xf32>
    %35 = linalg.fill ins(%cst : f32) outs(%34 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %36 = linalg.batch_matmul ins(%23, %33 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %cst_288 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_304 = tensor.expand_shape %37 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %38 = tensor.empty() : tensor<1x2x8x64xf32>
    %transposed_305 = linalg.transpose ins(%expanded_304 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_306 = linalg.transpose ins(%cst_287 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %39 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_306 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %40 = linalg.batch_matmul ins(%23, %39 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %cst_286 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_307 = tensor.expand_shape %41 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_308 = linalg.transpose ins(%expanded_307 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_309 = tensor.expand_shape %13 [[0], [1, 2], [3]] output_shape [1, 1, 8, 64] : tensor<1x8x64xf32> into tensor<1x1x8x64xf32>
    %expanded_310 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [1, 1, 8, 64] : tensor<1x8x64xf32> into tensor<1x1x8x64xf32>
    %42 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_302, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice = tensor.extract_slice %transposed_302[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_311 = tensor.extract_slice %transposed_302[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %43 = tensor.empty() : tensor<1x14x8x32xf32>
    %44 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_311 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_312 = tensor.concat dim(3) %44, %extracted_slice : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %45 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_312, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %46 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %45 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %47 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_305, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_313 = tensor.extract_slice %transposed_305[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_314 = tensor.extract_slice %transposed_305[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %48 = tensor.empty() : tensor<1x2x8x32xf32>
    %49 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_314 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_315 = tensor.concat dim(3) %49, %extracted_slice_313 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %50 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_315, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %51 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %52 = tensor.empty() : tensor<1x2x7x8x64xf32>
    %53 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%51 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed = tensor.collapse_shape %53 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %54 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_308 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %55 = tensor.empty() : tensor<1x14x64x8xf32>
    %transposed_316 = linalg.transpose ins(%collapsed : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_317 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_318 = tensor.collapse_shape %transposed_316 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %56 = tensor.empty() : tensor<14x8x8xf32>
    %57 = linalg.fill ins(%cst : f32) outs(%56 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %58 = linalg.batch_matmul ins(%collapsed_317, %collapsed_318 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_319 = tensor.expand_shape %58 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %59 = tensor.empty() : tensor<1x14x8x8xf32>
    %60 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_319 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %61 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %62 = tensor.empty() : tensor<1x14x8xi64>
    %63 = linalg.fill ins(%c0_i64 : i64) outs(%62 : tensor<1x14x8xi64>) -> tensor<1x14x8xi64>
    %64 = tensor.empty() : tensor<1x14x8xf32>
    %65 = linalg.fill ins(%cst_0 : f32) outs(%64 : tensor<1x14x8xf32>) -> tensor<1x14x8xf32>
    %66:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%61 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_320 = tensor.expand_shape %66#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %67 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %expanded_320 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %68 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %69 = tensor.empty() : tensor<1x14x8x1xf32>
    %70 = linalg.fill ins(%cst : f32) outs(%69 : tensor<1x14x8x1xf32>) -> tensor<1x14x8x1xf32>
    %71 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%68 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %72 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %71 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_321 = tensor.collapse_shape %72 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_322 = tensor.collapse_shape %54 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %73 = tensor.empty() : tensor<14x8x64xf32>
    %74 = linalg.fill ins(%cst : f32) outs(%73 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %75 = linalg.batch_matmul ins(%collapsed_321, %collapsed_322 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_323 = tensor.expand_shape %75 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %76 = tensor.empty() : tensor<1x8x14x64xf32>
    %transposed_324 = linalg.transpose ins(%expanded_323 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_325 = tensor.collapse_shape %transposed_324 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_326 = linalg.transpose ins(%cst_285 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %77 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_326 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %78 = linalg.batch_matmul ins(%collapsed_325, %77 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %79 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %78 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %80 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %81 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %82 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %83 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %84 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %85 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %84 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %86 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_284, %85 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %87 = tensor.empty() : tensor<896x4864xf32>
    %transposed_327 = linalg.transpose ins(%cst_283 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %88 = tensor.empty() : tensor<1x896x4864xf32>
    %89 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_327 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %90 = tensor.empty() : tensor<1x8x4864xf32>
    %91 = linalg.fill ins(%cst : f32) outs(%90 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %92 = linalg.batch_matmul ins(%86, %89 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %93 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %94 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %92 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_328 = linalg.transpose ins(%cst_282 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %95 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_328 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %96 = linalg.batch_matmul ins(%86, %95 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %97 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %96 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %98 = tensor.empty() : tensor<4864x896xf32>
    %transposed_329 = linalg.transpose ins(%cst_281 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %99 = tensor.empty() : tensor<1x4864x896xf32>
    %100 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_329 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %101 = linalg.batch_matmul ins(%97, %100 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %102 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %101 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %103 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %104 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%103 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %105 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %106 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %107 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %108 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %107 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %109 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_280, %108 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_330 = linalg.transpose ins(%cst_279 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %110 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_330 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %111 = linalg.batch_matmul ins(%109, %110 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %112 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%111, %cst_278 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_331 = tensor.expand_shape %112 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_332 = linalg.transpose ins(%expanded_331 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_333 = linalg.transpose ins(%cst_277 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %113 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_333 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %114 = linalg.batch_matmul ins(%109, %113 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %115 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %cst_276 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_334 = tensor.expand_shape %115 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_335 = linalg.transpose ins(%expanded_334 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_336 = linalg.transpose ins(%cst_275 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %116 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_336 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %117 = linalg.batch_matmul ins(%109, %116 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %118 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %cst_274 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_337 = tensor.expand_shape %118 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_338 = linalg.transpose ins(%expanded_337 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %119 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_332, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_339 = tensor.extract_slice %transposed_332[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_340 = tensor.extract_slice %transposed_332[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %120 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_340 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_341 = tensor.concat dim(3) %120, %extracted_slice_339 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %121 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_341, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %122 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %123 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_335, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_342 = tensor.extract_slice %transposed_335[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_343 = tensor.extract_slice %transposed_335[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %124 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_343 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_344 = tensor.concat dim(3) %124, %extracted_slice_342 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %125 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_344, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %126 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123, %125 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %127 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%126 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_345 = tensor.collapse_shape %127 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %128 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_338 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_346 = linalg.transpose ins(%collapsed_345 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_347 = tensor.collapse_shape %122 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_348 = tensor.collapse_shape %transposed_346 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %129 = linalg.batch_matmul ins(%collapsed_347, %collapsed_348 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_349 = tensor.expand_shape %129 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %130 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_349 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %131 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %132:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%131 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_350 = tensor.expand_shape %132#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %133 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131, %expanded_350 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %134 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %135 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%134 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %136 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134, %135 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_351 = tensor.collapse_shape %136 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_352 = tensor.collapse_shape %128 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %137 = linalg.batch_matmul ins(%collapsed_351, %collapsed_352 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_353 = tensor.expand_shape %137 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_354 = linalg.transpose ins(%expanded_353 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_355 = tensor.collapse_shape %transposed_354 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_356 = linalg.transpose ins(%cst_273 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %138 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_356 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %139 = linalg.batch_matmul ins(%collapsed_355, %138 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %140 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %139 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %141 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %142 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%141 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %143 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %144 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%143 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %145 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %146 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %145 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %147 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_272, %146 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_357 = linalg.transpose ins(%cst_271 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %148 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_357 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %149 = linalg.batch_matmul ins(%147, %148 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %150 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%149 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %151 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%150, %149 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_358 = linalg.transpose ins(%cst_270 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %152 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_358 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %153 = linalg.batch_matmul ins(%147, %152 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %154 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%151, %153 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_359 = linalg.transpose ins(%cst_269 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %155 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_359 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %156 = linalg.batch_matmul ins(%154, %155 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %157 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %156 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %158 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %159 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%158 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %160 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%159 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %161 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %162 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %163 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157, %162 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %164 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_268, %163 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_360 = linalg.transpose ins(%cst_267 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %165 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_360 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %166 = linalg.batch_matmul ins(%164, %165 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %167 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%166, %cst_266 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_361 = tensor.expand_shape %167 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_362 = linalg.transpose ins(%expanded_361 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_363 = linalg.transpose ins(%cst_265 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %168 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_363 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %169 = linalg.batch_matmul ins(%164, %168 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %170 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169, %cst_264 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_364 = tensor.expand_shape %170 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_365 = linalg.transpose ins(%expanded_364 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_366 = linalg.transpose ins(%cst_263 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %171 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_366 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %172 = linalg.batch_matmul ins(%164, %171 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %173 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172, %cst_262 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_367 = tensor.expand_shape %173 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_368 = linalg.transpose ins(%expanded_367 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %174 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_362, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_369 = tensor.extract_slice %transposed_362[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_370 = tensor.extract_slice %transposed_362[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %175 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_370 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_371 = tensor.concat dim(3) %175, %extracted_slice_369 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %176 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_371, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %177 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %176 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %178 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_365, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_372 = tensor.extract_slice %transposed_365[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_373 = tensor.extract_slice %transposed_365[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %179 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_373 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_374 = tensor.concat dim(3) %179, %extracted_slice_372 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %180 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_374, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %181 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178, %180 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %182 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%181 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_375 = tensor.collapse_shape %182 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %183 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_368 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_376 = linalg.transpose ins(%collapsed_375 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_377 = tensor.collapse_shape %177 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_378 = tensor.collapse_shape %transposed_376 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %184 = linalg.batch_matmul ins(%collapsed_377, %collapsed_378 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_379 = tensor.expand_shape %184 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %185 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_379 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %186 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %187:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%186 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_380 = tensor.expand_shape %187#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %188 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186, %expanded_380 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %189 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%188 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %190 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%189 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %191 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189, %190 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_381 = tensor.collapse_shape %191 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_382 = tensor.collapse_shape %183 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %192 = linalg.batch_matmul ins(%collapsed_381, %collapsed_382 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_383 = tensor.expand_shape %192 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_384 = linalg.transpose ins(%expanded_383 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_385 = tensor.collapse_shape %transposed_384 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_386 = linalg.transpose ins(%cst_261 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %193 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_386 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %194 = linalg.batch_matmul ins(%collapsed_385, %193 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %195 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157, %194 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %196 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %197 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%196 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %198 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%197 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %199 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %200 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%199 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %201 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195, %200 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %202 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_260, %201 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_387 = linalg.transpose ins(%cst_259 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %203 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_387 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %204 = linalg.batch_matmul ins(%202, %203 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %205 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %206 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%205, %204 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_388 = linalg.transpose ins(%cst_258 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %207 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_388 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %208 = linalg.batch_matmul ins(%202, %207 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %209 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%206, %208 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_389 = linalg.transpose ins(%cst_257 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %210 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_389 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %211 = linalg.batch_matmul ins(%209, %210 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %212 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195, %211 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %213 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%212 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %214 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%213 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %215 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %216 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%215 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %217 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%216 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %218 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%212, %217 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %219 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_256, %218 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_390 = linalg.transpose ins(%cst_255 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %220 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_390 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %221 = linalg.batch_matmul ins(%219, %220 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %222 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%221, %cst_254 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_391 = tensor.expand_shape %222 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_392 = linalg.transpose ins(%expanded_391 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_393 = linalg.transpose ins(%cst_253 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %223 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_393 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %224 = linalg.batch_matmul ins(%219, %223 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %225 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%224, %cst_252 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_394 = tensor.expand_shape %225 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_395 = linalg.transpose ins(%expanded_394 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_396 = linalg.transpose ins(%cst_251 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %226 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_396 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %227 = linalg.batch_matmul ins(%219, %226 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %228 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227, %cst_250 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_397 = tensor.expand_shape %228 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_398 = linalg.transpose ins(%expanded_397 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %229 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_392, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_399 = tensor.extract_slice %transposed_392[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_400 = tensor.extract_slice %transposed_392[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %230 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_400 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_401 = tensor.concat dim(3) %230, %extracted_slice_399 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %231 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_401, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %232 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%229, %231 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %233 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_395, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_402 = tensor.extract_slice %transposed_395[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_403 = tensor.extract_slice %transposed_395[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %234 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_403 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_404 = tensor.concat dim(3) %234, %extracted_slice_402 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %235 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_404, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %236 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%233, %235 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %237 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%236 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_405 = tensor.collapse_shape %237 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %238 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_398 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_406 = linalg.transpose ins(%collapsed_405 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_407 = tensor.collapse_shape %232 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_408 = tensor.collapse_shape %transposed_406 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %239 = linalg.batch_matmul ins(%collapsed_407, %collapsed_408 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_409 = tensor.expand_shape %239 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %240 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_409 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %241 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%240, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %242:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%241 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_410 = tensor.expand_shape %242#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %243 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241, %expanded_410 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %244 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%243 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %245 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%244 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %246 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244, %245 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_411 = tensor.collapse_shape %246 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_412 = tensor.collapse_shape %238 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %247 = linalg.batch_matmul ins(%collapsed_411, %collapsed_412 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_413 = tensor.expand_shape %247 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_414 = linalg.transpose ins(%expanded_413 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_415 = tensor.collapse_shape %transposed_414 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_416 = linalg.transpose ins(%cst_249 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %248 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_416 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %249 = linalg.batch_matmul ins(%collapsed_415, %248 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %250 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%212, %249 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %251 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %252 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%251 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %253 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %254 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%253 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %255 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %256 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250, %255 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %257 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_248, %256 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_417 = linalg.transpose ins(%cst_247 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %258 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_417 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %259 = linalg.batch_matmul ins(%257, %258 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %260 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %261 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%260, %259 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_418 = linalg.transpose ins(%cst_246 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %262 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_418 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %263 = linalg.batch_matmul ins(%257, %262 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %264 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%261, %263 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_419 = linalg.transpose ins(%cst_245 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %265 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_419 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %266 = linalg.batch_matmul ins(%264, %265 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %267 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250, %266 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %268 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%267 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %269 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%268 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %270 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%269 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %271 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%270 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %272 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%271 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %273 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%267, %272 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %274 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_244, %273 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_420 = linalg.transpose ins(%cst_243 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %275 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_420 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %276 = linalg.batch_matmul ins(%274, %275 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %277 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%276, %cst_242 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_421 = tensor.expand_shape %277 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_422 = linalg.transpose ins(%expanded_421 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_423 = linalg.transpose ins(%cst_241 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %278 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_423 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %279 = linalg.batch_matmul ins(%274, %278 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %280 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%279, %cst_240 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_424 = tensor.expand_shape %280 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_425 = linalg.transpose ins(%expanded_424 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_426 = linalg.transpose ins(%cst_239 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %281 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_426 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %282 = linalg.batch_matmul ins(%274, %281 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %283 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%282, %cst_238 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_427 = tensor.expand_shape %283 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_428 = linalg.transpose ins(%expanded_427 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %284 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_422, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_429 = tensor.extract_slice %transposed_422[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_430 = tensor.extract_slice %transposed_422[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %285 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_430 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_431 = tensor.concat dim(3) %285, %extracted_slice_429 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %286 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_431, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %287 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%284, %286 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %288 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_425, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_432 = tensor.extract_slice %transposed_425[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_433 = tensor.extract_slice %transposed_425[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %289 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_433 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_434 = tensor.concat dim(3) %289, %extracted_slice_432 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %290 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_434, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %291 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%288, %290 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %292 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%291 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_435 = tensor.collapse_shape %292 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %293 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_428 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_436 = linalg.transpose ins(%collapsed_435 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_437 = tensor.collapse_shape %287 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_438 = tensor.collapse_shape %transposed_436 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %294 = linalg.batch_matmul ins(%collapsed_437, %collapsed_438 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_439 = tensor.expand_shape %294 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %295 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_439 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %296 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%295, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %297:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%296 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_440 = tensor.expand_shape %297#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %298 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296, %expanded_440 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %299 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %300 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%299 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %301 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %300 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_441 = tensor.collapse_shape %301 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_442 = tensor.collapse_shape %293 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %302 = linalg.batch_matmul ins(%collapsed_441, %collapsed_442 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_443 = tensor.expand_shape %302 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_444 = linalg.transpose ins(%expanded_443 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_445 = tensor.collapse_shape %transposed_444 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_446 = linalg.transpose ins(%cst_237 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %303 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_446 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %304 = linalg.batch_matmul ins(%collapsed_445, %303 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %305 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%267, %304 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %306 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%305 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %307 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%306 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %308 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%307 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %309 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%308 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %310 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%309 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %311 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%305, %310 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %312 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_236, %311 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_447 = linalg.transpose ins(%cst_235 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %313 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_447 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %314 = linalg.batch_matmul ins(%312, %313 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %315 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%314 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %316 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%315, %314 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_448 = linalg.transpose ins(%cst_234 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %317 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_448 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %318 = linalg.batch_matmul ins(%312, %317 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %319 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%316, %318 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_449 = linalg.transpose ins(%cst_233 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %320 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_449 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %321 = linalg.batch_matmul ins(%319, %320 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %322 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%305, %321 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %323 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %324 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%323 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %325 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%324 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %326 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%325 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %327 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%326 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %328 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322, %327 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %329 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_232, %328 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_450 = linalg.transpose ins(%cst_231 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %330 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_450 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %331 = linalg.batch_matmul ins(%329, %330 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %332 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%331, %cst_230 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_451 = tensor.expand_shape %332 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_452 = linalg.transpose ins(%expanded_451 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_453 = linalg.transpose ins(%cst_229 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %333 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_453 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %334 = linalg.batch_matmul ins(%329, %333 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %335 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%334, %cst_228 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_454 = tensor.expand_shape %335 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_455 = linalg.transpose ins(%expanded_454 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_456 = linalg.transpose ins(%cst_227 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %336 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_456 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %337 = linalg.batch_matmul ins(%329, %336 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %338 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337, %cst_226 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_457 = tensor.expand_shape %338 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_458 = linalg.transpose ins(%expanded_457 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %339 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_452, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_459 = tensor.extract_slice %transposed_452[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_460 = tensor.extract_slice %transposed_452[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %340 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_460 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_461 = tensor.concat dim(3) %340, %extracted_slice_459 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %341 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_461, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %342 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%339, %341 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %343 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_455, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_462 = tensor.extract_slice %transposed_455[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_463 = tensor.extract_slice %transposed_455[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %344 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_463 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_464 = tensor.concat dim(3) %344, %extracted_slice_462 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %345 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_464, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %346 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %345 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %347 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%346 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_465 = tensor.collapse_shape %347 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %348 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_458 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_466 = linalg.transpose ins(%collapsed_465 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_467 = tensor.collapse_shape %342 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_468 = tensor.collapse_shape %transposed_466 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %349 = linalg.batch_matmul ins(%collapsed_467, %collapsed_468 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_469 = tensor.expand_shape %349 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %350 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_469 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %351 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%350, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %352:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%351 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_470 = tensor.expand_shape %352#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %353 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%351, %expanded_470 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %354 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %355 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%354 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %356 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354, %355 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_471 = tensor.collapse_shape %356 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_472 = tensor.collapse_shape %348 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %357 = linalg.batch_matmul ins(%collapsed_471, %collapsed_472 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_473 = tensor.expand_shape %357 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_474 = linalg.transpose ins(%expanded_473 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_475 = tensor.collapse_shape %transposed_474 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_476 = linalg.transpose ins(%cst_225 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %358 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_476 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %359 = linalg.batch_matmul ins(%collapsed_475, %358 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %360 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322, %359 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %361 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%360 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %362 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%361 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %363 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%362 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %364 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%363 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %365 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%364 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %366 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%360, %365 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %367 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_224, %366 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_477 = linalg.transpose ins(%cst_223 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %368 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_477 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %369 = linalg.batch_matmul ins(%367, %368 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %370 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%369 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %371 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%370, %369 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_478 = linalg.transpose ins(%cst_222 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %372 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_478 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %373 = linalg.batch_matmul ins(%367, %372 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %374 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%371, %373 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_479 = linalg.transpose ins(%cst_221 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %375 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_479 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %376 = linalg.batch_matmul ins(%374, %375 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %377 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%360, %376 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %378 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%377 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %379 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%378 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %380 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%379 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %381 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%380 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %382 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%381 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %383 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%377, %382 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %384 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_220, %383 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_480 = linalg.transpose ins(%cst_219 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %385 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_480 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %386 = linalg.batch_matmul ins(%384, %385 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %387 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%386, %cst_218 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_481 = tensor.expand_shape %387 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_482 = linalg.transpose ins(%expanded_481 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_483 = linalg.transpose ins(%cst_217 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %388 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_483 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %389 = linalg.batch_matmul ins(%384, %388 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %390 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%389, %cst_216 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_484 = tensor.expand_shape %390 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_485 = linalg.transpose ins(%expanded_484 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_486 = linalg.transpose ins(%cst_215 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %391 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_486 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %392 = linalg.batch_matmul ins(%384, %391 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %393 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392, %cst_214 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_487 = tensor.expand_shape %393 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_488 = linalg.transpose ins(%expanded_487 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %394 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_482, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_489 = tensor.extract_slice %transposed_482[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_490 = tensor.extract_slice %transposed_482[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %395 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_490 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_491 = tensor.concat dim(3) %395, %extracted_slice_489 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %396 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_491, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %397 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%394, %396 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %398 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_485, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_492 = tensor.extract_slice %transposed_485[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_493 = tensor.extract_slice %transposed_485[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %399 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_493 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_494 = tensor.concat dim(3) %399, %extracted_slice_492 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %400 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_494, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %401 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%398, %400 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %402 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%401 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_495 = tensor.collapse_shape %402 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %403 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_488 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_496 = linalg.transpose ins(%collapsed_495 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_497 = tensor.collapse_shape %397 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_498 = tensor.collapse_shape %transposed_496 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %404 = linalg.batch_matmul ins(%collapsed_497, %collapsed_498 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_499 = tensor.expand_shape %404 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %405 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_499 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %406 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%405, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %407:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%406 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_500 = tensor.expand_shape %407#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %408 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %expanded_500 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %409 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %410 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%409 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %411 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %410 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_501 = tensor.collapse_shape %411 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_502 = tensor.collapse_shape %403 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %412 = linalg.batch_matmul ins(%collapsed_501, %collapsed_502 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_503 = tensor.expand_shape %412 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_504 = linalg.transpose ins(%expanded_503 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_505 = tensor.collapse_shape %transposed_504 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_506 = linalg.transpose ins(%cst_213 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %413 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_506 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %414 = linalg.batch_matmul ins(%collapsed_505, %413 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %415 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%377, %414 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %416 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%415 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %417 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%416 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %418 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%417 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %419 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%418 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %420 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%419 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %421 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%415, %420 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %422 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_212, %421 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_507 = linalg.transpose ins(%cst_211 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %423 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_507 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %424 = linalg.batch_matmul ins(%422, %423 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %425 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%424 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %426 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425, %424 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_508 = linalg.transpose ins(%cst_210 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %427 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_508 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %428 = linalg.batch_matmul ins(%422, %427 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %429 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%426, %428 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_509 = linalg.transpose ins(%cst_209 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %430 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_509 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %431 = linalg.batch_matmul ins(%429, %430 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %432 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%415, %431 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %433 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%432 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %434 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%433 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %435 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%434 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %436 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%435 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %437 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%436 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %438 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%432, %437 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %439 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_208, %438 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_510 = linalg.transpose ins(%cst_207 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %440 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_510 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %441 = linalg.batch_matmul ins(%439, %440 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %442 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%441, %cst_206 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_511 = tensor.expand_shape %442 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_512 = linalg.transpose ins(%expanded_511 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_513 = linalg.transpose ins(%cst_205 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %443 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_513 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %444 = linalg.batch_matmul ins(%439, %443 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %445 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%444, %cst_204 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_514 = tensor.expand_shape %445 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_515 = linalg.transpose ins(%expanded_514 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_516 = linalg.transpose ins(%cst_203 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %446 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_516 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %447 = linalg.batch_matmul ins(%439, %446 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %448 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%447, %cst_202 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_517 = tensor.expand_shape %448 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_518 = linalg.transpose ins(%expanded_517 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %449 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_512, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_519 = tensor.extract_slice %transposed_512[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_520 = tensor.extract_slice %transposed_512[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %450 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_520 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_521 = tensor.concat dim(3) %450, %extracted_slice_519 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %451 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_521, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %452 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449, %451 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %453 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_515, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_522 = tensor.extract_slice %transposed_515[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_523 = tensor.extract_slice %transposed_515[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %454 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_523 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_524 = tensor.concat dim(3) %454, %extracted_slice_522 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %455 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_524, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %456 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%453, %455 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %457 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%456 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_525 = tensor.collapse_shape %457 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %458 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_518 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_526 = linalg.transpose ins(%collapsed_525 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_527 = tensor.collapse_shape %452 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_528 = tensor.collapse_shape %transposed_526 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %459 = linalg.batch_matmul ins(%collapsed_527, %collapsed_528 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_529 = tensor.expand_shape %459 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %460 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_529 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %461 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %462:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%461 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_530 = tensor.expand_shape %462#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %463 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%461, %expanded_530 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %464 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %465 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%464 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %466 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%464, %465 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_531 = tensor.collapse_shape %466 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_532 = tensor.collapse_shape %458 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %467 = linalg.batch_matmul ins(%collapsed_531, %collapsed_532 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_533 = tensor.expand_shape %467 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_534 = linalg.transpose ins(%expanded_533 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_535 = tensor.collapse_shape %transposed_534 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_536 = linalg.transpose ins(%cst_201 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %468 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_536 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %469 = linalg.batch_matmul ins(%collapsed_535, %468 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %470 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%432, %469 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %471 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %472 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%471 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %473 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%472 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %474 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%473 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %475 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%474 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %476 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470, %475 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %477 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_200, %476 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_537 = linalg.transpose ins(%cst_199 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %478 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_537 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %479 = linalg.batch_matmul ins(%477, %478 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %480 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %481 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%480, %479 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_538 = linalg.transpose ins(%cst_198 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %482 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_538 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %483 = linalg.batch_matmul ins(%477, %482 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %484 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481, %483 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_539 = linalg.transpose ins(%cst_197 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %485 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_539 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %486 = linalg.batch_matmul ins(%484, %485 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %487 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470, %486 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %488 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%487 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %489 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%488 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %490 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%489 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %491 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%490 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %492 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%491 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %493 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%487, %492 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %494 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_196, %493 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_540 = linalg.transpose ins(%cst_195 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %495 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_540 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %496 = linalg.batch_matmul ins(%494, %495 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %497 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%496, %cst_194 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_541 = tensor.expand_shape %497 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_542 = linalg.transpose ins(%expanded_541 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_543 = linalg.transpose ins(%cst_193 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %498 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_543 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %499 = linalg.batch_matmul ins(%494, %498 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %500 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%499, %cst_192 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_544 = tensor.expand_shape %500 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_545 = linalg.transpose ins(%expanded_544 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_546 = linalg.transpose ins(%cst_191 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %501 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_546 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %502 = linalg.batch_matmul ins(%494, %501 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %503 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%502, %cst_190 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_547 = tensor.expand_shape %503 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_548 = linalg.transpose ins(%expanded_547 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %504 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_542, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_549 = tensor.extract_slice %transposed_542[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_550 = tensor.extract_slice %transposed_542[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %505 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_550 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_551 = tensor.concat dim(3) %505, %extracted_slice_549 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %506 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_551, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %507 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %506 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %508 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_545, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_552 = tensor.extract_slice %transposed_545[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_553 = tensor.extract_slice %transposed_545[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %509 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_553 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_554 = tensor.concat dim(3) %509, %extracted_slice_552 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %510 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_554, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %511 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%508, %510 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %512 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%511 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_555 = tensor.collapse_shape %512 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %513 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_548 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_556 = linalg.transpose ins(%collapsed_555 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_557 = tensor.collapse_shape %507 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_558 = tensor.collapse_shape %transposed_556 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %514 = linalg.batch_matmul ins(%collapsed_557, %collapsed_558 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_559 = tensor.expand_shape %514 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %515 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_559 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %516 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%515, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %517:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%516 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_560 = tensor.expand_shape %517#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %518 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516, %expanded_560 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %519 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%518 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %520 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%519 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %521 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%519, %520 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_561 = tensor.collapse_shape %521 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_562 = tensor.collapse_shape %513 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %522 = linalg.batch_matmul ins(%collapsed_561, %collapsed_562 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_563 = tensor.expand_shape %522 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_564 = linalg.transpose ins(%expanded_563 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_565 = tensor.collapse_shape %transposed_564 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_566 = linalg.transpose ins(%cst_189 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %523 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_566 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %524 = linalg.batch_matmul ins(%collapsed_565, %523 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %525 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%487, %524 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %526 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%525 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %527 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%526 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %528 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%527 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %529 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%528 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %530 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%529 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %531 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%525, %530 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %532 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_188, %531 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_567 = linalg.transpose ins(%cst_187 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %533 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_567 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %534 = linalg.batch_matmul ins(%532, %533 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %535 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%534 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %536 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%535, %534 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_568 = linalg.transpose ins(%cst_186 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %537 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_568 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %538 = linalg.batch_matmul ins(%532, %537 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %539 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%536, %538 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_569 = linalg.transpose ins(%cst_185 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %540 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_569 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %541 = linalg.batch_matmul ins(%539, %540 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %542 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%525, %541 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %543 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %544 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%543 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %545 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %546 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %547 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%546 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %548 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542, %547 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %549 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_184, %548 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_570 = linalg.transpose ins(%cst_183 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %550 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_570 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %551 = linalg.batch_matmul ins(%549, %550 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %552 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%551, %cst_182 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_571 = tensor.expand_shape %552 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_572 = linalg.transpose ins(%expanded_571 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_573 = linalg.transpose ins(%cst_181 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %553 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_573 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %554 = linalg.batch_matmul ins(%549, %553 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %555 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%554, %cst_180 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_574 = tensor.expand_shape %555 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_575 = linalg.transpose ins(%expanded_574 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_576 = linalg.transpose ins(%cst_179 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %556 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_576 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %557 = linalg.batch_matmul ins(%549, %556 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %558 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%557, %cst_178 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_577 = tensor.expand_shape %558 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_578 = linalg.transpose ins(%expanded_577 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %559 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_572, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_579 = tensor.extract_slice %transposed_572[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_580 = tensor.extract_slice %transposed_572[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %560 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_580 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_581 = tensor.concat dim(3) %560, %extracted_slice_579 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %561 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_581, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %562 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%559, %561 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %563 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_575, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_582 = tensor.extract_slice %transposed_575[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_583 = tensor.extract_slice %transposed_575[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %564 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_583 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_584 = tensor.concat dim(3) %564, %extracted_slice_582 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %565 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_584, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %566 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%563, %565 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %567 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%566 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_585 = tensor.collapse_shape %567 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %568 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_578 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_586 = linalg.transpose ins(%collapsed_585 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_587 = tensor.collapse_shape %562 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_588 = tensor.collapse_shape %transposed_586 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %569 = linalg.batch_matmul ins(%collapsed_587, %collapsed_588 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_589 = tensor.expand_shape %569 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %570 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_589 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %571 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%570, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %572:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%571 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_590 = tensor.expand_shape %572#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %573 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%571, %expanded_590 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %574 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%573 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %575 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%574 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %576 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%574, %575 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_591 = tensor.collapse_shape %576 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_592 = tensor.collapse_shape %568 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %577 = linalg.batch_matmul ins(%collapsed_591, %collapsed_592 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_593 = tensor.expand_shape %577 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_594 = linalg.transpose ins(%expanded_593 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_595 = tensor.collapse_shape %transposed_594 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_596 = linalg.transpose ins(%cst_177 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %578 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_596 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %579 = linalg.batch_matmul ins(%collapsed_595, %578 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %580 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542, %579 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %581 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%580 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %582 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%581 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %583 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%582 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %584 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%583 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %585 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%584 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %586 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%580, %585 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %587 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_176, %586 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_597 = linalg.transpose ins(%cst_175 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %588 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_597 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %589 = linalg.batch_matmul ins(%587, %588 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %590 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%589 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %591 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%590, %589 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_598 = linalg.transpose ins(%cst_174 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %592 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_598 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %593 = linalg.batch_matmul ins(%587, %592 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %594 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%591, %593 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_599 = linalg.transpose ins(%cst_173 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %595 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_599 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %596 = linalg.batch_matmul ins(%594, %595 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %597 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%580, %596 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %598 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %599 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%598 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %600 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %601 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%600 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %602 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%601 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %603 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597, %602 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %604 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_172, %603 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_600 = linalg.transpose ins(%cst_171 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %605 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_600 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %606 = linalg.batch_matmul ins(%604, %605 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %607 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%606, %cst_170 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_601 = tensor.expand_shape %607 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_602 = linalg.transpose ins(%expanded_601 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_603 = linalg.transpose ins(%cst_169 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %608 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_603 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %609 = linalg.batch_matmul ins(%604, %608 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %610 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%609, %cst_168 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_604 = tensor.expand_shape %610 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_605 = linalg.transpose ins(%expanded_604 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_606 = linalg.transpose ins(%cst_167 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %611 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_606 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %612 = linalg.batch_matmul ins(%604, %611 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %613 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%612, %cst_166 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_607 = tensor.expand_shape %613 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_608 = linalg.transpose ins(%expanded_607 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %614 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_602, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_609 = tensor.extract_slice %transposed_602[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_610 = tensor.extract_slice %transposed_602[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %615 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_610 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_611 = tensor.concat dim(3) %615, %extracted_slice_609 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %616 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_611, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %617 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%614, %616 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %618 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_605, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_612 = tensor.extract_slice %transposed_605[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_613 = tensor.extract_slice %transposed_605[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %619 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_613 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_614 = tensor.concat dim(3) %619, %extracted_slice_612 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %620 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_614, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %621 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %620 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %622 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%621 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_615 = tensor.collapse_shape %622 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %623 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_608 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_616 = linalg.transpose ins(%collapsed_615 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_617 = tensor.collapse_shape %617 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_618 = tensor.collapse_shape %transposed_616 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %624 = linalg.batch_matmul ins(%collapsed_617, %collapsed_618 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_619 = tensor.expand_shape %624 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %625 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_619 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %626 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%625, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %627:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%626 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_620 = tensor.expand_shape %627#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %628 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%626, %expanded_620 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %629 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%628 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %630 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%629 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %631 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%629, %630 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_621 = tensor.collapse_shape %631 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_622 = tensor.collapse_shape %623 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %632 = linalg.batch_matmul ins(%collapsed_621, %collapsed_622 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_623 = tensor.expand_shape %632 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_624 = linalg.transpose ins(%expanded_623 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_625 = tensor.collapse_shape %transposed_624 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_626 = linalg.transpose ins(%cst_165 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %633 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_626 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %634 = linalg.batch_matmul ins(%collapsed_625, %633 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %635 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597, %634 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %636 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%635 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %637 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%636 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %638 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%637 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %639 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%638 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %640 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%639 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %641 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%635, %640 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %642 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_164, %641 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_627 = linalg.transpose ins(%cst_163 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %643 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_627 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %644 = linalg.batch_matmul ins(%642, %643 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %645 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%644 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %646 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%645, %644 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_628 = linalg.transpose ins(%cst_162 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %647 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_628 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %648 = linalg.batch_matmul ins(%642, %647 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %649 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%646, %648 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_629 = linalg.transpose ins(%cst_161 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %650 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_629 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %651 = linalg.batch_matmul ins(%649, %650 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %652 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%635, %651 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %653 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%652 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %654 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%653 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %655 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%654 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %656 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%655 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %657 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%656 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %658 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%652, %657 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %659 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_160, %658 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_630 = linalg.transpose ins(%cst_159 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %660 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_630 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %661 = linalg.batch_matmul ins(%659, %660 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %662 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%661, %cst_158 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_631 = tensor.expand_shape %662 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_632 = linalg.transpose ins(%expanded_631 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_633 = linalg.transpose ins(%cst_157 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %663 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_633 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %664 = linalg.batch_matmul ins(%659, %663 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %665 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%664, %cst_156 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_634 = tensor.expand_shape %665 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_635 = linalg.transpose ins(%expanded_634 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_636 = linalg.transpose ins(%cst_155 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %666 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_636 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %667 = linalg.batch_matmul ins(%659, %666 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %668 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%667, %cst_154 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_637 = tensor.expand_shape %668 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_638 = linalg.transpose ins(%expanded_637 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %669 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_632, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_639 = tensor.extract_slice %transposed_632[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_640 = tensor.extract_slice %transposed_632[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %670 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_640 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_641 = tensor.concat dim(3) %670, %extracted_slice_639 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %671 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_641, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %672 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%669, %671 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %673 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_635, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_642 = tensor.extract_slice %transposed_635[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_643 = tensor.extract_slice %transposed_635[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %674 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_643 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_644 = tensor.concat dim(3) %674, %extracted_slice_642 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %675 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_644, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %676 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%673, %675 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %677 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%676 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_645 = tensor.collapse_shape %677 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %678 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_638 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_646 = linalg.transpose ins(%collapsed_645 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_647 = tensor.collapse_shape %672 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_648 = tensor.collapse_shape %transposed_646 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %679 = linalg.batch_matmul ins(%collapsed_647, %collapsed_648 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_649 = tensor.expand_shape %679 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %680 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_649 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %681 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%680, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %682:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%681 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_650 = tensor.expand_shape %682#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %683 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%681, %expanded_650 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %684 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%683 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %685 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%684 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %686 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%684, %685 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_651 = tensor.collapse_shape %686 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_652 = tensor.collapse_shape %678 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %687 = linalg.batch_matmul ins(%collapsed_651, %collapsed_652 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_653 = tensor.expand_shape %687 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_654 = linalg.transpose ins(%expanded_653 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_655 = tensor.collapse_shape %transposed_654 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_656 = linalg.transpose ins(%cst_153 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %688 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_656 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %689 = linalg.batch_matmul ins(%collapsed_655, %688 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %690 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%652, %689 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %691 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%690 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %692 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%691 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %693 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%692 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %694 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%693 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %695 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%694 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %696 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%690, %695 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %697 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_152, %696 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_657 = linalg.transpose ins(%cst_151 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %698 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_657 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %699 = linalg.batch_matmul ins(%697, %698 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %700 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%699 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %701 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%700, %699 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_658 = linalg.transpose ins(%cst_150 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %702 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_658 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %703 = linalg.batch_matmul ins(%697, %702 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %704 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%701, %703 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_659 = linalg.transpose ins(%cst_149 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %705 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_659 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %706 = linalg.batch_matmul ins(%704, %705 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %707 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%690, %706 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %708 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %709 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%708 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %710 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%709 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %711 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%710 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %712 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%711 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %713 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707, %712 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %714 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_148, %713 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_660 = linalg.transpose ins(%cst_147 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %715 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_660 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %716 = linalg.batch_matmul ins(%714, %715 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %717 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716, %cst_146 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_661 = tensor.expand_shape %717 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_662 = linalg.transpose ins(%expanded_661 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_663 = linalg.transpose ins(%cst_145 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %718 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_663 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %719 = linalg.batch_matmul ins(%714, %718 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %720 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%719, %cst_144 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_664 = tensor.expand_shape %720 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_665 = linalg.transpose ins(%expanded_664 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_666 = linalg.transpose ins(%cst_143 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %721 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_666 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %722 = linalg.batch_matmul ins(%714, %721 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %723 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722, %cst_142 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_667 = tensor.expand_shape %723 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_668 = linalg.transpose ins(%expanded_667 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %724 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_662, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_669 = tensor.extract_slice %transposed_662[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_670 = tensor.extract_slice %transposed_662[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %725 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_670 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_671 = tensor.concat dim(3) %725, %extracted_slice_669 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %726 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_671, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %727 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%724, %726 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %728 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_665, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_672 = tensor.extract_slice %transposed_665[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_673 = tensor.extract_slice %transposed_665[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %729 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_673 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_674 = tensor.concat dim(3) %729, %extracted_slice_672 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %730 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_674, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %731 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%728, %730 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %732 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%731 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_675 = tensor.collapse_shape %732 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %733 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_668 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_676 = linalg.transpose ins(%collapsed_675 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_677 = tensor.collapse_shape %727 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_678 = tensor.collapse_shape %transposed_676 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %734 = linalg.batch_matmul ins(%collapsed_677, %collapsed_678 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_679 = tensor.expand_shape %734 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %735 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_679 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %736 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%735, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %737:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%736 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_680 = tensor.expand_shape %737#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %738 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%736, %expanded_680 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %739 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%738 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %740 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%739 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %741 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%739, %740 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_681 = tensor.collapse_shape %741 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_682 = tensor.collapse_shape %733 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %742 = linalg.batch_matmul ins(%collapsed_681, %collapsed_682 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_683 = tensor.expand_shape %742 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_684 = linalg.transpose ins(%expanded_683 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_685 = tensor.collapse_shape %transposed_684 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_686 = linalg.transpose ins(%cst_141 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %743 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_686 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %744 = linalg.batch_matmul ins(%collapsed_685, %743 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %745 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707, %744 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %746 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%745 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %747 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%746 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %748 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%747 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %749 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%748 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %750 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%749 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %751 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%745, %750 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %752 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_140, %751 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_687 = linalg.transpose ins(%cst_139 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %753 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_687 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %754 = linalg.batch_matmul ins(%752, %753 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %755 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%754 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %756 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%755, %754 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_688 = linalg.transpose ins(%cst_138 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %757 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_688 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %758 = linalg.batch_matmul ins(%752, %757 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %759 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%756, %758 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_689 = linalg.transpose ins(%cst_137 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %760 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_689 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %761 = linalg.batch_matmul ins(%759, %760 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %762 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%745, %761 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %763 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%762 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %764 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%763 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %765 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %766 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%765 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %767 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%766 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %768 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%762, %767 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %769 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_136, %768 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_690 = linalg.transpose ins(%cst_135 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %770 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_690 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %771 = linalg.batch_matmul ins(%769, %770 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %772 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%771, %cst_134 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_691 = tensor.expand_shape %772 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_692 = linalg.transpose ins(%expanded_691 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_693 = linalg.transpose ins(%cst_133 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %773 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_693 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %774 = linalg.batch_matmul ins(%769, %773 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %775 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%774, %cst_132 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_694 = tensor.expand_shape %775 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_695 = linalg.transpose ins(%expanded_694 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_696 = linalg.transpose ins(%cst_131 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %776 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_696 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %777 = linalg.batch_matmul ins(%769, %776 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %778 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%777, %cst_130 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_697 = tensor.expand_shape %778 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_698 = linalg.transpose ins(%expanded_697 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %779 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_692, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_699 = tensor.extract_slice %transposed_692[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_700 = tensor.extract_slice %transposed_692[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %780 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_700 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_701 = tensor.concat dim(3) %780, %extracted_slice_699 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %781 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_701, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %782 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%779, %781 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %783 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_695, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_702 = tensor.extract_slice %transposed_695[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_703 = tensor.extract_slice %transposed_695[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %784 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_703 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_704 = tensor.concat dim(3) %784, %extracted_slice_702 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %785 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_704, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %786 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%783, %785 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %787 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%786 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_705 = tensor.collapse_shape %787 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %788 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_698 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_706 = linalg.transpose ins(%collapsed_705 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_707 = tensor.collapse_shape %782 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_708 = tensor.collapse_shape %transposed_706 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %789 = linalg.batch_matmul ins(%collapsed_707, %collapsed_708 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_709 = tensor.expand_shape %789 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %790 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_709 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %791 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%790, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %792:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%791 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_710 = tensor.expand_shape %792#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %793 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%791, %expanded_710 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %794 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%793 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %795 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%794 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %796 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%794, %795 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_711 = tensor.collapse_shape %796 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_712 = tensor.collapse_shape %788 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %797 = linalg.batch_matmul ins(%collapsed_711, %collapsed_712 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_713 = tensor.expand_shape %797 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_714 = linalg.transpose ins(%expanded_713 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_715 = tensor.collapse_shape %transposed_714 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_716 = linalg.transpose ins(%cst_129 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %798 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_716 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %799 = linalg.batch_matmul ins(%collapsed_715, %798 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %800 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%762, %799 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %801 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%800 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %802 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%801 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %803 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%802 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %804 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%803 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %805 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%804 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %806 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%800, %805 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %807 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_128, %806 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_717 = linalg.transpose ins(%cst_127 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %808 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_717 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %809 = linalg.batch_matmul ins(%807, %808 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %810 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%809 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %811 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%810, %809 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_718 = linalg.transpose ins(%cst_126 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %812 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_718 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %813 = linalg.batch_matmul ins(%807, %812 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %814 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%811, %813 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_719 = linalg.transpose ins(%cst_125 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %815 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_719 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %816 = linalg.batch_matmul ins(%814, %815 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %817 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%800, %816 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %818 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%817 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %819 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%818 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %820 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%819 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %821 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%820 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %822 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%821 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %823 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%817, %822 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %824 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_124, %823 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_720 = linalg.transpose ins(%cst_123 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %825 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_720 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %826 = linalg.batch_matmul ins(%824, %825 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %827 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%826, %cst_122 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_721 = tensor.expand_shape %827 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_722 = linalg.transpose ins(%expanded_721 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_723 = linalg.transpose ins(%cst_121 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %828 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_723 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %829 = linalg.batch_matmul ins(%824, %828 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %830 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%829, %cst_120 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_724 = tensor.expand_shape %830 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_725 = linalg.transpose ins(%expanded_724 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_726 = linalg.transpose ins(%cst_119 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %831 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_726 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %832 = linalg.batch_matmul ins(%824, %831 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %833 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%832, %cst_118 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_727 = tensor.expand_shape %833 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_728 = linalg.transpose ins(%expanded_727 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %834 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_722, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_729 = tensor.extract_slice %transposed_722[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_730 = tensor.extract_slice %transposed_722[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %835 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_730 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_731 = tensor.concat dim(3) %835, %extracted_slice_729 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %836 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_731, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %837 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%834, %836 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %838 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_725, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_732 = tensor.extract_slice %transposed_725[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_733 = tensor.extract_slice %transposed_725[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %839 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_733 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_734 = tensor.concat dim(3) %839, %extracted_slice_732 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %840 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_734, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %841 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%838, %840 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %842 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%841 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_735 = tensor.collapse_shape %842 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %843 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_728 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_736 = linalg.transpose ins(%collapsed_735 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_737 = tensor.collapse_shape %837 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_738 = tensor.collapse_shape %transposed_736 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %844 = linalg.batch_matmul ins(%collapsed_737, %collapsed_738 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_739 = tensor.expand_shape %844 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %845 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_739 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %846 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%845, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %847:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%846 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_740 = tensor.expand_shape %847#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %848 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%846, %expanded_740 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %849 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%848 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %850 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%849 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %851 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%849, %850 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_741 = tensor.collapse_shape %851 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_742 = tensor.collapse_shape %843 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %852 = linalg.batch_matmul ins(%collapsed_741, %collapsed_742 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_743 = tensor.expand_shape %852 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_744 = linalg.transpose ins(%expanded_743 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_745 = tensor.collapse_shape %transposed_744 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_746 = linalg.transpose ins(%cst_117 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %853 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_746 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %854 = linalg.batch_matmul ins(%collapsed_745, %853 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %855 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%817, %854 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %856 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%855 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %857 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%856 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %858 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%857 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %859 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%858 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %860 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%859 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %861 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%855, %860 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %862 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_116, %861 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_747 = linalg.transpose ins(%cst_115 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %863 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_747 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %864 = linalg.batch_matmul ins(%862, %863 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %865 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%864 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %866 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%865, %864 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_748 = linalg.transpose ins(%cst_114 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %867 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_748 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %868 = linalg.batch_matmul ins(%862, %867 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %869 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%866, %868 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_749 = linalg.transpose ins(%cst_113 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %870 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_749 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %871 = linalg.batch_matmul ins(%869, %870 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %872 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%855, %871 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %873 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%872 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %874 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%873 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %875 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%874 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %876 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%875 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %877 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%876 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %878 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%872, %877 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %879 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_112, %878 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_750 = linalg.transpose ins(%cst_111 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %880 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_750 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %881 = linalg.batch_matmul ins(%879, %880 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %882 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%881, %cst_110 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_751 = tensor.expand_shape %882 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_752 = linalg.transpose ins(%expanded_751 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_753 = linalg.transpose ins(%cst_109 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %883 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_753 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %884 = linalg.batch_matmul ins(%879, %883 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %885 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%884, %cst_108 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_754 = tensor.expand_shape %885 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_755 = linalg.transpose ins(%expanded_754 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_756 = linalg.transpose ins(%cst_107 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %886 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_756 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %887 = linalg.batch_matmul ins(%879, %886 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %888 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%887, %cst_106 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_757 = tensor.expand_shape %888 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_758 = linalg.transpose ins(%expanded_757 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %889 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_752, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_759 = tensor.extract_slice %transposed_752[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_760 = tensor.extract_slice %transposed_752[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %890 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_760 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_761 = tensor.concat dim(3) %890, %extracted_slice_759 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %891 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_761, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %892 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%889, %891 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %893 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_755, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_762 = tensor.extract_slice %transposed_755[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_763 = tensor.extract_slice %transposed_755[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %894 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_763 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_764 = tensor.concat dim(3) %894, %extracted_slice_762 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %895 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_764, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %896 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%893, %895 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %897 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%896 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_765 = tensor.collapse_shape %897 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %898 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_758 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_766 = linalg.transpose ins(%collapsed_765 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_767 = tensor.collapse_shape %892 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_768 = tensor.collapse_shape %transposed_766 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %899 = linalg.batch_matmul ins(%collapsed_767, %collapsed_768 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_769 = tensor.expand_shape %899 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %900 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_769 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %901 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%900, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %902:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%901 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_770 = tensor.expand_shape %902#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %903 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%901, %expanded_770 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %904 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%903 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %905 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%904 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %906 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%904, %905 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_771 = tensor.collapse_shape %906 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_772 = tensor.collapse_shape %898 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %907 = linalg.batch_matmul ins(%collapsed_771, %collapsed_772 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_773 = tensor.expand_shape %907 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_774 = linalg.transpose ins(%expanded_773 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_775 = tensor.collapse_shape %transposed_774 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_776 = linalg.transpose ins(%cst_105 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %908 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_776 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %909 = linalg.batch_matmul ins(%collapsed_775, %908 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %910 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%872, %909 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %911 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%910 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %912 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%911 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %913 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%912 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %914 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%913 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %915 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%914 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %916 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%910, %915 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %917 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_104, %916 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_777 = linalg.transpose ins(%cst_103 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %918 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_777 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %919 = linalg.batch_matmul ins(%917, %918 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %920 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%919 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %921 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%920, %919 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_778 = linalg.transpose ins(%cst_102 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %922 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_778 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %923 = linalg.batch_matmul ins(%917, %922 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %924 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%921, %923 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_779 = linalg.transpose ins(%cst_101 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %925 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_779 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %926 = linalg.batch_matmul ins(%924, %925 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %927 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%910, %926 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %928 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%927 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %929 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%928 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %930 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%929 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %931 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%930 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %932 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%931 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %933 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%927, %932 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %934 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_100, %933 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_780 = linalg.transpose ins(%cst_99 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %935 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_780 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %936 = linalg.batch_matmul ins(%934, %935 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %937 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%936, %cst_98 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_781 = tensor.expand_shape %937 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_782 = linalg.transpose ins(%expanded_781 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_783 = linalg.transpose ins(%cst_97 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %938 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_783 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %939 = linalg.batch_matmul ins(%934, %938 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %940 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%939, %cst_96 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_784 = tensor.expand_shape %940 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_785 = linalg.transpose ins(%expanded_784 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_786 = linalg.transpose ins(%cst_95 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %941 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_786 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %942 = linalg.batch_matmul ins(%934, %941 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %943 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%942, %cst_94 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_787 = tensor.expand_shape %943 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_788 = linalg.transpose ins(%expanded_787 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %944 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_782, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_789 = tensor.extract_slice %transposed_782[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_790 = tensor.extract_slice %transposed_782[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %945 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_790 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_791 = tensor.concat dim(3) %945, %extracted_slice_789 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %946 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_791, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %947 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%944, %946 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %948 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_785, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_792 = tensor.extract_slice %transposed_785[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_793 = tensor.extract_slice %transposed_785[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %949 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_793 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_794 = tensor.concat dim(3) %949, %extracted_slice_792 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %950 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_794, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %951 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%948, %950 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %952 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%951 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_795 = tensor.collapse_shape %952 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %953 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_788 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_796 = linalg.transpose ins(%collapsed_795 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_797 = tensor.collapse_shape %947 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_798 = tensor.collapse_shape %transposed_796 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %954 = linalg.batch_matmul ins(%collapsed_797, %collapsed_798 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_799 = tensor.expand_shape %954 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %955 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_799 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %956 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%955, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %957:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%956 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_800 = tensor.expand_shape %957#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %958 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%956, %expanded_800 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %959 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%958 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %960 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%959 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %961 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%959, %960 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_801 = tensor.collapse_shape %961 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_802 = tensor.collapse_shape %953 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %962 = linalg.batch_matmul ins(%collapsed_801, %collapsed_802 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_803 = tensor.expand_shape %962 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_804 = linalg.transpose ins(%expanded_803 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_805 = tensor.collapse_shape %transposed_804 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_806 = linalg.transpose ins(%cst_93 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %963 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_806 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %964 = linalg.batch_matmul ins(%collapsed_805, %963 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %965 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%927, %964 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %966 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %967 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%966 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %968 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%967 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %969 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%968 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %970 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%969 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %971 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965, %970 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %972 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_92, %971 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_807 = linalg.transpose ins(%cst_91 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %973 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_807 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %974 = linalg.batch_matmul ins(%972, %973 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %975 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %976 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%975, %974 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_808 = linalg.transpose ins(%cst_90 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %977 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_808 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %978 = linalg.batch_matmul ins(%972, %977 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %979 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%976, %978 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_809 = linalg.transpose ins(%cst_89 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %980 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_809 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %981 = linalg.batch_matmul ins(%979, %980 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %982 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965, %981 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %983 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%982 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %984 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%983 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %985 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%984 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %986 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%985 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %987 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%986 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %988 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%982, %987 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %989 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_88, %988 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_810 = linalg.transpose ins(%cst_87 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %990 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_810 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %991 = linalg.batch_matmul ins(%989, %990 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %992 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991, %cst_86 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_811 = tensor.expand_shape %992 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_812 = linalg.transpose ins(%expanded_811 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_813 = linalg.transpose ins(%cst_85 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %993 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_813 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %994 = linalg.batch_matmul ins(%989, %993 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %995 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%994, %cst_84 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_814 = tensor.expand_shape %995 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_815 = linalg.transpose ins(%expanded_814 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_816 = linalg.transpose ins(%cst_83 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %996 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_816 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %997 = linalg.batch_matmul ins(%989, %996 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %998 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%997, %cst_82 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_817 = tensor.expand_shape %998 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_818 = linalg.transpose ins(%expanded_817 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %999 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_812, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_819 = tensor.extract_slice %transposed_812[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_820 = tensor.extract_slice %transposed_812[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1000 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_820 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_821 = tensor.concat dim(3) %1000, %extracted_slice_819 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1001 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_821, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1002 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%999, %1001 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1003 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_815, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_822 = tensor.extract_slice %transposed_815[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_823 = tensor.extract_slice %transposed_815[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1004 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_823 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_824 = tensor.concat dim(3) %1004, %extracted_slice_822 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1005 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_824, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1006 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1003, %1005 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1007 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1006 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_825 = tensor.collapse_shape %1007 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1008 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_818 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_826 = linalg.transpose ins(%collapsed_825 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_827 = tensor.collapse_shape %1002 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_828 = tensor.collapse_shape %transposed_826 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1009 = linalg.batch_matmul ins(%collapsed_827, %collapsed_828 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_829 = tensor.expand_shape %1009 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1010 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_829 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1011 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1010, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1012:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1011 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_830 = tensor.expand_shape %1012#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1013 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1011, %expanded_830 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1014 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1013 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1015 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1014 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1016 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1014, %1015 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_831 = tensor.collapse_shape %1016 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_832 = tensor.collapse_shape %1008 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1017 = linalg.batch_matmul ins(%collapsed_831, %collapsed_832 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_833 = tensor.expand_shape %1017 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_834 = linalg.transpose ins(%expanded_833 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_835 = tensor.collapse_shape %transposed_834 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_836 = linalg.transpose ins(%cst_81 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1018 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_836 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1019 = linalg.batch_matmul ins(%collapsed_835, %1018 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1020 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%982, %1019 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1021 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1020 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1022 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1021 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1023 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1022 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1024 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1023 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1025 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1024 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1026 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1020, %1025 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1027 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_80, %1026 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_837 = linalg.transpose ins(%cst_79 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1028 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_837 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1029 = linalg.batch_matmul ins(%1027, %1028 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1030 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1029 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1031 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1030, %1029 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_838 = linalg.transpose ins(%cst_78 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1032 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_838 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1033 = linalg.batch_matmul ins(%1027, %1032 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1034 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1031, %1033 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_839 = linalg.transpose ins(%cst_77 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1035 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_839 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1036 = linalg.batch_matmul ins(%1034, %1035 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1037 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1020, %1036 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1038 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1039 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1038 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1040 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1039 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1041 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1040 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1042 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1041 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1043 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037, %1042 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1044 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_76, %1043 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_840 = linalg.transpose ins(%cst_75 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1045 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_840 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1046 = linalg.batch_matmul ins(%1044, %1045 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1047 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1046, %cst_74 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_841 = tensor.expand_shape %1047 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_842 = linalg.transpose ins(%expanded_841 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_843 = linalg.transpose ins(%cst_73 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1048 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_843 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1049 = linalg.batch_matmul ins(%1044, %1048 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1050 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1049, %cst_72 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_844 = tensor.expand_shape %1050 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_845 = linalg.transpose ins(%expanded_844 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_846 = linalg.transpose ins(%cst_71 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1051 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_846 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1052 = linalg.batch_matmul ins(%1044, %1051 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1053 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1052, %cst_70 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_847 = tensor.expand_shape %1053 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_848 = linalg.transpose ins(%expanded_847 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %1054 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_842, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_849 = tensor.extract_slice %transposed_842[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_850 = tensor.extract_slice %transposed_842[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1055 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_850 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_851 = tensor.concat dim(3) %1055, %extracted_slice_849 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1056 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_851, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1057 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1054, %1056 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1058 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_845, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_852 = tensor.extract_slice %transposed_845[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_853 = tensor.extract_slice %transposed_845[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1059 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_853 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_854 = tensor.concat dim(3) %1059, %extracted_slice_852 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1060 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_854, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1061 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1058, %1060 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1062 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1061 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_855 = tensor.collapse_shape %1062 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1063 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_848 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_856 = linalg.transpose ins(%collapsed_855 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_857 = tensor.collapse_shape %1057 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_858 = tensor.collapse_shape %transposed_856 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1064 = linalg.batch_matmul ins(%collapsed_857, %collapsed_858 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_859 = tensor.expand_shape %1064 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1065 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_859 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1066 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1065, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1067:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1066 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_860 = tensor.expand_shape %1067#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1068 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1066, %expanded_860 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1069 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1068 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1070 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1069 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1071 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1069, %1070 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_861 = tensor.collapse_shape %1071 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_862 = tensor.collapse_shape %1063 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1072 = linalg.batch_matmul ins(%collapsed_861, %collapsed_862 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_863 = tensor.expand_shape %1072 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_864 = linalg.transpose ins(%expanded_863 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_865 = tensor.collapse_shape %transposed_864 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_866 = linalg.transpose ins(%cst_69 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1073 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_866 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1074 = linalg.batch_matmul ins(%collapsed_865, %1073 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1075 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037, %1074 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1076 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1075 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1077 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1076 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1078 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1077 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1079 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1078 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1080 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1079 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1081 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1075, %1080 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1082 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_68, %1081 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_867 = linalg.transpose ins(%cst_67 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1083 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_867 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1084 = linalg.batch_matmul ins(%1082, %1083 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1085 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1084 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1086 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1085, %1084 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_868 = linalg.transpose ins(%cst_66 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1087 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_868 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1088 = linalg.batch_matmul ins(%1082, %1087 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1089 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1086, %1088 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_869 = linalg.transpose ins(%cst_65 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1090 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_869 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1091 = linalg.batch_matmul ins(%1089, %1090 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1092 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1075, %1091 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1093 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1092 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1094 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1093 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1095 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1094 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1096 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1095 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1097 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1096 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1098 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1092, %1097 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1099 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_64, %1098 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_870 = linalg.transpose ins(%cst_63 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1100 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_870 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1101 = linalg.batch_matmul ins(%1099, %1100 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1102 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1101, %cst_62 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_871 = tensor.expand_shape %1102 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_872 = linalg.transpose ins(%expanded_871 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_873 = linalg.transpose ins(%cst_61 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1103 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_873 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1104 = linalg.batch_matmul ins(%1099, %1103 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1105 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1104, %cst_60 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_874 = tensor.expand_shape %1105 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_875 = linalg.transpose ins(%expanded_874 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_876 = linalg.transpose ins(%cst_59 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1106 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_876 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1107 = linalg.batch_matmul ins(%1099, %1106 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1108 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1107, %cst_58 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_877 = tensor.expand_shape %1108 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_878 = linalg.transpose ins(%expanded_877 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %1109 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_872, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_879 = tensor.extract_slice %transposed_872[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_880 = tensor.extract_slice %transposed_872[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1110 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_880 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_881 = tensor.concat dim(3) %1110, %extracted_slice_879 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1111 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_881, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1112 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1109, %1111 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1113 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_875, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_882 = tensor.extract_slice %transposed_875[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_883 = tensor.extract_slice %transposed_875[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1114 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_883 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_884 = tensor.concat dim(3) %1114, %extracted_slice_882 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1115 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_884, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1116 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1113, %1115 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1117 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1116 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_885 = tensor.collapse_shape %1117 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1118 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_878 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_886 = linalg.transpose ins(%collapsed_885 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_887 = tensor.collapse_shape %1112 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_888 = tensor.collapse_shape %transposed_886 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1119 = linalg.batch_matmul ins(%collapsed_887, %collapsed_888 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_889 = tensor.expand_shape %1119 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1120 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_889 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1121 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1120, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1122:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1121 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_890 = tensor.expand_shape %1122#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1123 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1121, %expanded_890 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1124 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1123 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1125 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1124 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1126 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1124, %1125 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_891 = tensor.collapse_shape %1126 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_892 = tensor.collapse_shape %1118 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1127 = linalg.batch_matmul ins(%collapsed_891, %collapsed_892 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_893 = tensor.expand_shape %1127 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_894 = linalg.transpose ins(%expanded_893 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_895 = tensor.collapse_shape %transposed_894 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_896 = linalg.transpose ins(%cst_57 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1128 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_896 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1129 = linalg.batch_matmul ins(%collapsed_895, %1128 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1130 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1092, %1129 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1131 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1130 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1132 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1131 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1133 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1132 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1134 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1133 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1135 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1134 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1136 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1130, %1135 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1137 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_56, %1136 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_897 = linalg.transpose ins(%cst_55 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1138 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_897 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1139 = linalg.batch_matmul ins(%1137, %1138 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1140 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1139 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1141 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1140, %1139 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_898 = linalg.transpose ins(%cst_54 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1142 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_898 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1143 = linalg.batch_matmul ins(%1137, %1142 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1144 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1141, %1143 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_899 = linalg.transpose ins(%cst_53 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1145 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_899 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1146 = linalg.batch_matmul ins(%1144, %1145 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1147 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1130, %1146 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1148 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1149 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1148 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1150 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1149 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1151 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1150 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1152 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1151 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1153 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %1152 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1154 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_52, %1153 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_900 = linalg.transpose ins(%cst_51 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1155 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_900 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1156 = linalg.batch_matmul ins(%1154, %1155 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1157 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1156, %cst_50 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_901 = tensor.expand_shape %1157 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_902 = linalg.transpose ins(%expanded_901 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_903 = linalg.transpose ins(%cst_49 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1158 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_903 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1159 = linalg.batch_matmul ins(%1154, %1158 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1160 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1159, %cst_48 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_904 = tensor.expand_shape %1160 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_905 = linalg.transpose ins(%expanded_904 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_906 = linalg.transpose ins(%cst_47 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1161 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_906 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1162 = linalg.batch_matmul ins(%1154, %1161 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1163 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1162, %cst_46 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_907 = tensor.expand_shape %1163 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_908 = linalg.transpose ins(%expanded_907 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %1164 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_902, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_909 = tensor.extract_slice %transposed_902[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_910 = tensor.extract_slice %transposed_902[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1165 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_910 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_911 = tensor.concat dim(3) %1165, %extracted_slice_909 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1166 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_911, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1167 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1164, %1166 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1168 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_905, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_912 = tensor.extract_slice %transposed_905[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_913 = tensor.extract_slice %transposed_905[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1169 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_913 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_914 = tensor.concat dim(3) %1169, %extracted_slice_912 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1170 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_914, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1171 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1168, %1170 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1172 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1171 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_915 = tensor.collapse_shape %1172 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1173 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_908 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_916 = linalg.transpose ins(%collapsed_915 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_917 = tensor.collapse_shape %1167 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_918 = tensor.collapse_shape %transposed_916 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1174 = linalg.batch_matmul ins(%collapsed_917, %collapsed_918 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_919 = tensor.expand_shape %1174 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1175 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_919 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1176 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1175, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1177:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1176 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_920 = tensor.expand_shape %1177#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1178 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1176, %expanded_920 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1179 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1178 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1180 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1179 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1181 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1179, %1180 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_921 = tensor.collapse_shape %1181 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_922 = tensor.collapse_shape %1173 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1182 = linalg.batch_matmul ins(%collapsed_921, %collapsed_922 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_923 = tensor.expand_shape %1182 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_924 = linalg.transpose ins(%expanded_923 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_925 = tensor.collapse_shape %transposed_924 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_926 = linalg.transpose ins(%cst_45 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1183 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_926 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1184 = linalg.batch_matmul ins(%collapsed_925, %1183 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1185 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %1184 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1186 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1185 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1187 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1186 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1188 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1187 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1189 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1188 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1190 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1189 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1191 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1185, %1190 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1192 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_44, %1191 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_927 = linalg.transpose ins(%cst_43 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1193 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_927 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1194 = linalg.batch_matmul ins(%1192, %1193 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1195 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1194 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1196 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1195, %1194 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_928 = linalg.transpose ins(%cst_42 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1197 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_928 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1198 = linalg.batch_matmul ins(%1192, %1197 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1199 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1196, %1198 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_929 = linalg.transpose ins(%cst_41 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1200 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_929 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1201 = linalg.batch_matmul ins(%1199, %1200 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1202 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1185, %1201 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1203 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1202 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1204 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1203 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1205 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1204 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1206 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1205 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1207 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1206 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1208 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1202, %1207 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1209 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_40, %1208 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_930 = linalg.transpose ins(%cst_39 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1210 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_930 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1211 = linalg.batch_matmul ins(%1209, %1210 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1212 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1211, %cst_38 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_931 = tensor.expand_shape %1212 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_932 = linalg.transpose ins(%expanded_931 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_933 = linalg.transpose ins(%cst_37 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1213 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_933 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1214 = linalg.batch_matmul ins(%1209, %1213 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1215 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1214, %cst_36 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_934 = tensor.expand_shape %1215 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_935 = linalg.transpose ins(%expanded_934 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_936 = linalg.transpose ins(%cst_35 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1216 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_936 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1217 = linalg.batch_matmul ins(%1209, %1216 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1218 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1217, %cst_34 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_937 = tensor.expand_shape %1218 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_938 = linalg.transpose ins(%expanded_937 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %1219 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_932, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_939 = tensor.extract_slice %transposed_932[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_940 = tensor.extract_slice %transposed_932[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1220 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_940 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_941 = tensor.concat dim(3) %1220, %extracted_slice_939 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1221 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_941, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1222 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1219, %1221 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1223 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_935, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_942 = tensor.extract_slice %transposed_935[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_943 = tensor.extract_slice %transposed_935[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1224 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_943 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_944 = tensor.concat dim(3) %1224, %extracted_slice_942 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1225 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_944, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1226 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1223, %1225 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1227 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1226 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_945 = tensor.collapse_shape %1227 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1228 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_938 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_946 = linalg.transpose ins(%collapsed_945 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_947 = tensor.collapse_shape %1222 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_948 = tensor.collapse_shape %transposed_946 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1229 = linalg.batch_matmul ins(%collapsed_947, %collapsed_948 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_949 = tensor.expand_shape %1229 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1230 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_949 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1231 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1230, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1232:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1231 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_950 = tensor.expand_shape %1232#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1233 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1231, %expanded_950 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1234 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1233 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1235 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1234 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1236 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1234, %1235 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_951 = tensor.collapse_shape %1236 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_952 = tensor.collapse_shape %1228 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1237 = linalg.batch_matmul ins(%collapsed_951, %collapsed_952 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_953 = tensor.expand_shape %1237 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_954 = linalg.transpose ins(%expanded_953 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_955 = tensor.collapse_shape %transposed_954 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_956 = linalg.transpose ins(%cst_33 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1238 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_956 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1239 = linalg.batch_matmul ins(%collapsed_955, %1238 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1240 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1202, %1239 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1241 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1240 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1242 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1241 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1243 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1242 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1244 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1243 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1245 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1244 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1246 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1240, %1245 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1247 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_32, %1246 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_957 = linalg.transpose ins(%cst_31 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1248 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_957 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1249 = linalg.batch_matmul ins(%1247, %1248 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1250 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1249 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1251 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1250, %1249 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_958 = linalg.transpose ins(%cst_30 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1252 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_958 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1253 = linalg.batch_matmul ins(%1247, %1252 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1254 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1251, %1253 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_959 = linalg.transpose ins(%cst_29 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1255 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_959 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1256 = linalg.batch_matmul ins(%1254, %1255 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1257 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1240, %1256 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1258 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1257 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1259 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1258 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1260 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1259 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1261 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1260 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1262 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1261 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1263 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1257, %1262 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1264 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_28, %1263 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_960 = linalg.transpose ins(%cst_27 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1265 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_960 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1266 = linalg.batch_matmul ins(%1264, %1265 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1267 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1266, %cst_26 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_961 = tensor.expand_shape %1267 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_962 = linalg.transpose ins(%expanded_961 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_963 = linalg.transpose ins(%cst_25 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1268 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_963 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1269 = linalg.batch_matmul ins(%1264, %1268 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1270 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1269, %cst_24 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_964 = tensor.expand_shape %1270 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_965 = linalg.transpose ins(%expanded_964 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_966 = linalg.transpose ins(%cst_23 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1271 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_966 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1272 = linalg.batch_matmul ins(%1264, %1271 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1273 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1272, %cst_22 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_967 = tensor.expand_shape %1273 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_968 = linalg.transpose ins(%expanded_967 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %1274 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_962, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_969 = tensor.extract_slice %transposed_962[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_970 = tensor.extract_slice %transposed_962[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1275 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_970 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_971 = tensor.concat dim(3) %1275, %extracted_slice_969 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1276 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_971, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1277 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1274, %1276 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1278 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_965, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_972 = tensor.extract_slice %transposed_965[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_973 = tensor.extract_slice %transposed_965[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1279 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_973 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_974 = tensor.concat dim(3) %1279, %extracted_slice_972 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1280 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_974, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1281 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1278, %1280 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1282 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1281 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_975 = tensor.collapse_shape %1282 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1283 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_968 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_976 = linalg.transpose ins(%collapsed_975 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_977 = tensor.collapse_shape %1277 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_978 = tensor.collapse_shape %transposed_976 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1284 = linalg.batch_matmul ins(%collapsed_977, %collapsed_978 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_979 = tensor.expand_shape %1284 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1285 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_979 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1286 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1285, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1287:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1286 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_980 = tensor.expand_shape %1287#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1288 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1286, %expanded_980 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1289 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1288 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1290 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1289 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1291 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1289, %1290 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_981 = tensor.collapse_shape %1291 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_982 = tensor.collapse_shape %1283 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1292 = linalg.batch_matmul ins(%collapsed_981, %collapsed_982 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_983 = tensor.expand_shape %1292 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_984 = linalg.transpose ins(%expanded_983 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_985 = tensor.collapse_shape %transposed_984 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_986 = linalg.transpose ins(%cst_21 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1293 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_986 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1294 = linalg.batch_matmul ins(%collapsed_985, %1293 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1295 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1257, %1294 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1296 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1297 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1296 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1298 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1297 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1299 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1298 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1300 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1299 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1301 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295, %1300 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1302 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_20, %1301 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_987 = linalg.transpose ins(%cst_19 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1303 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_987 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1304 = linalg.batch_matmul ins(%1302, %1303 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1305 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1304 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1306 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1305, %1304 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_988 = linalg.transpose ins(%cst_18 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1307 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_988 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1308 = linalg.batch_matmul ins(%1302, %1307 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1309 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306, %1308 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_989 = linalg.transpose ins(%cst_17 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1310 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_989 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1311 = linalg.batch_matmul ins(%1309, %1310 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1312 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295, %1311 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1313 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1312 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1314 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1313 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1315 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1314 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1316 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1315 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1317 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1318 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1312, %1317 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1319 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_16, %1318 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_990 = linalg.transpose ins(%cst_15 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1320 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_990 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1321 = linalg.batch_matmul ins(%1319, %1320 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1322 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1321, %cst_14 : tensor<1x8x896xf32>, tensor<896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %expanded_991 = tensor.expand_shape %1322 [[0], [1], [2, 3]] output_shape [1, 8, 14, 64] : tensor<1x8x896xf32> into tensor<1x8x14x64xf32>
    %transposed_992 = linalg.transpose ins(%expanded_991 : tensor<1x8x14x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_993 = linalg.transpose ins(%cst_13 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1323 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_993 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1324 = linalg.batch_matmul ins(%1319, %1323 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1325 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1324, %cst_12 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_994 = tensor.expand_shape %1325 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_995 = linalg.transpose ins(%expanded_994 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_996 = linalg.transpose ins(%cst_11 : tensor<128x896xf32>) outs(%31 : tensor<896x128xf32>) permutation = [1, 0] 
    %1326 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_996 : tensor<896x128xf32>) outs(%32 : tensor<1x896x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x128xf32>
    %1327 = linalg.batch_matmul ins(%1319, %1326 : tensor<1x8x896xf32>, tensor<1x896x128xf32>) outs(%35 : tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
    %1328 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1327, %cst_10 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%34 : tensor<1x8x128xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x128xf32>
    %expanded_997 = tensor.expand_shape %1328 [[0], [1], [2, 3]] output_shape [1, 8, 2, 64] : tensor<1x8x128xf32> into tensor<1x8x2x64xf32>
    %transposed_998 = linalg.transpose ins(%expanded_997 : tensor<1x8x2x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) permutation = [0, 2, 1, 3] 
    %1329 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_992, %expanded_309 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %extracted_slice_999 = tensor.extract_slice %transposed_992[0, 0, 0, 0] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %extracted_slice_1000 = tensor.extract_slice %transposed_992[0, 0, 0, 32] [1, 14, 8, 32] [1, 1, 1, 1] : tensor<1x14x8x64xf32> to tensor<1x14x8x32xf32>
    %1330 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1000 : tensor<1x14x8x32xf32>) outs(%43 : tensor<1x14x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x32xf32>
    %concat_1001 = tensor.concat dim(3) %1330, %extracted_slice_999 : (tensor<1x14x8x32xf32>, tensor<1x14x8x32xf32>) -> tensor<1x14x8x64xf32>
    %1331 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1001, %expanded_310 : tensor<1x14x8x64xf32>, tensor<1x1x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1332 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1329, %1331 : tensor<1x14x8x64xf32>, tensor<1x14x8x64xf32>) outs(%30 : tensor<1x14x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x64xf32>
    %1333 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_995, %expanded_309 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %extracted_slice_1002 = tensor.extract_slice %transposed_995[0, 0, 0, 0] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %extracted_slice_1003 = tensor.extract_slice %transposed_995[0, 0, 0, 32] [1, 2, 8, 32] [1, 1, 1, 1] : tensor<1x2x8x64xf32> to tensor<1x2x8x32xf32>
    %1334 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1003 : tensor<1x2x8x32xf32>) outs(%48 : tensor<1x2x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x32xf32>
    %concat_1004 = tensor.concat dim(3) %1334, %extracted_slice_1002 : (tensor<1x2x8x32xf32>, tensor<1x2x8x32xf32>) -> tensor<1x2x8x64xf32>
    %1335 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_1004, %expanded_310 : tensor<1x2x8x64xf32>, tensor<1x1x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1336 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1333, %1335 : tensor<1x2x8x64xf32>, tensor<1x2x8x64xf32>) outs(%38 : tensor<1x2x8x64xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x2x8x64xf32>
    %1337 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1336 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %collapsed_1005 = tensor.collapse_shape %1337 [[0], [1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<1x14x8x64xf32>
    %1338 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_998 : tensor<1x2x8x64xf32>) outs(%52 : tensor<1x2x7x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x7x8x64xf32>
    %transposed_1006 = linalg.transpose ins(%collapsed_1005 : tensor<1x14x8x64xf32>) outs(%55 : tensor<1x14x64x8xf32>) permutation = [0, 1, 3, 2] 
    %collapsed_1007 = tensor.collapse_shape %1332 [[0, 1], [2], [3]] : tensor<1x14x8x64xf32> into tensor<14x8x64xf32>
    %collapsed_1008 = tensor.collapse_shape %transposed_1006 [[0, 1], [2], [3]] : tensor<1x14x64x8xf32> into tensor<14x64x8xf32>
    %1339 = linalg.batch_matmul ins(%collapsed_1007, %collapsed_1008 : tensor<14x8x64xf32>, tensor<14x64x8xf32>) outs(%57 : tensor<14x8x8xf32>) -> tensor<14x8x8xf32>
    %expanded_1009 = tensor.expand_shape %1339 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf32> into tensor<1x14x8x8xf32>
    %1340 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1009 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.mulf %in, %cst_297 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1341 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1340, %7 : tensor<1x14x8x8xf32>, tensor<1x1x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1342:2 = linalg.generic {indexing_maps = [#map5, #map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1341 : tensor<1x14x8x8xf32>) outs(%65, %63 : tensor<1x14x8xf32>, tensor<1x14x8xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1021: i64):
      %1381 = linalg.index 3 : index
      %1382 = arith.index_cast %1381 : index to i64
      %1383 = arith.maximumf %in, %out : f32
      %1384 = arith.cmpf ogt, %in, %out : f32
      %1385 = arith.select %1384, %1382, %out_1021 : i64
      linalg.yield %1383, %1385 : f32, i64
    } -> (tensor<1x14x8xf32>, tensor<1x14x8xi64>)
    %expanded_1010 = tensor.expand_shape %1342#0 [[0], [1], [2, 3]] output_shape [1, 14, 8, 1] : tensor<1x14x8xf32> into tensor<1x14x8x1xf32>
    %1343 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1341, %expanded_1010 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.subf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1344 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1343 : tensor<1x14x8x8xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.exp %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %1345 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1344 : tensor<1x14x8x8xf32>) outs(%70 : tensor<1x14x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x1xf32>
    %1346 = linalg.generic {indexing_maps = [#map5, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1344, %1345 : tensor<1x14x8x8xf32>, tensor<1x14x8x1xf32>) outs(%59 : tensor<1x14x8x8xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.divf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x14x8x8xf32>
    %collapsed_1011 = tensor.collapse_shape %1346 [[0, 1], [2], [3]] : tensor<1x14x8x8xf32> into tensor<14x8x8xf32>
    %collapsed_1012 = tensor.collapse_shape %1338 [[0, 1, 2], [3], [4]] : tensor<1x2x7x8x64xf32> into tensor<14x8x64xf32>
    %1347 = linalg.batch_matmul ins(%collapsed_1011, %collapsed_1012 : tensor<14x8x8xf32>, tensor<14x8x64xf32>) outs(%74 : tensor<14x8x64xf32>) -> tensor<14x8x64xf32>
    %expanded_1013 = tensor.expand_shape %1347 [[0, 1], [2], [3]] output_shape [1, 14, 8, 64] : tensor<14x8x64xf32> into tensor<1x14x8x64xf32>
    %transposed_1014 = linalg.transpose ins(%expanded_1013 : tensor<1x14x8x64xf32>) outs(%76 : tensor<1x8x14x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_1015 = tensor.collapse_shape %transposed_1014 [[0], [1], [2, 3]] : tensor<1x8x14x64xf32> into tensor<1x8x896xf32>
    %transposed_1016 = linalg.transpose ins(%cst_9 : tensor<896x896xf32>) outs(%24 : tensor<896x896xf32>) permutation = [1, 0] 
    %1348 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1016 : tensor<896x896xf32>) outs(%25 : tensor<1x896x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x896xf32>
    %1349 = linalg.batch_matmul ins(%collapsed_1015, %1348 : tensor<1x8x896xf32>, tensor<1x896x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1350 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1312, %1349 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1351 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1350 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1352 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1351 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1353 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1352 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1354 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1353 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1355 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1354 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1356 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1350, %1355 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1357 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_8, %1356 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %transposed_1017 = linalg.transpose ins(%cst_7 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1358 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1017 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1359 = linalg.batch_matmul ins(%1357, %1358 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1360 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1359 : tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.negf %in : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1382, %cst_1 : f32
      %1384 = arith.divf %cst_1, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x4864xf32>
    %1361 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1360, %1359 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_1018 = linalg.transpose ins(%cst_6 : tensor<4864x896xf32>) outs(%87 : tensor<896x4864xf32>) permutation = [1, 0] 
    %1362 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1018 : tensor<896x4864xf32>) outs(%88 : tensor<1x896x4864xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x4864xf32>
    %1363 = linalg.batch_matmul ins(%1357, %1362 : tensor<1x8x896xf32>, tensor<1x896x4864xf32>) outs(%91 : tensor<1x8x4864xf32>) -> tensor<1x8x4864xf32>
    %1364 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1361, %1363 : tensor<1x8x4864xf32>, tensor<1x8x4864xf32>) outs(%90 : tensor<1x8x4864xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x4864xf32>
    %transposed_1019 = linalg.transpose ins(%cst_5 : tensor<896x4864xf32>) outs(%98 : tensor<4864x896xf32>) permutation = [1, 0] 
    %1365 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1019 : tensor<4864x896xf32>) outs(%99 : tensor<1x4864x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4864x896xf32>
    %1366 = linalg.batch_matmul ins(%1364, %1365 : tensor<1x8x4864xf32>, tensor<1x4864x896xf32>) outs(%27 : tensor<1x8x896xf32>) -> tensor<1x8x896xf32>
    %1367 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1350, %1366 : tensor<1x8x896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.addf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1368 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1367 : tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1369 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1368 : tensor<1x8x896xf32>) outs(%17 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.addf %in, %out : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1370 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1369 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.divf %in, %cst_296 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1371 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1370 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = arith.truncf %cst_293 : f64 to f32
      %1382 = arith.addf %in, %1381 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1372 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1371 : tensor<1x8x1xf32>) outs(%16 : tensor<1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1381 = math.rsqrt %in : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x1xf32>
    %1373 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1367, %1372 : tensor<1x8x896xf32>, tensor<1x8x1xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1374 = linalg.generic {indexing_maps = [#map9, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_4, %1373 : tensor<896xf32>, tensor<1x8x896xf32>) outs(%2 : tensor<1x8x896xf32>) {
    ^bb0(%in: f32, %in_1021: f32, %out: f32):
      %1381 = arith.mulf %in, %in_1021 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x8x896xf32>
    %1375 = tensor.empty() : tensor<896x151936xf32>
    %transposed_1020 = linalg.transpose ins(%cst_295 : tensor<151936x896xf32>) outs(%1375 : tensor<896x151936xf32>) permutation = [1, 0] 
    %1376 = tensor.empty() : tensor<1x896x151936xf32>
    %1377 = linalg.generic {indexing_maps = [#map10, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1020 : tensor<896x151936xf32>) outs(%1376 : tensor<1x896x151936xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x896x151936xf32>
    %1378 = tensor.empty() : tensor<1x8x151936xf32>
    %1379 = linalg.fill ins(%cst : f32) outs(%1378 : tensor<1x8x151936xf32>) -> tensor<1x8x151936xf32>
    %1380 = linalg.batch_matmul ins(%1374, %1377 : tensor<1x8x896xf32>, tensor<1x896x151936xf32>) outs(%1379 : tensor<1x8x151936xf32>) -> tensor<1x8x151936xf32>
    return %1380 : tensor<1x8x151936xf32>
  }
}

{-#

#-}
