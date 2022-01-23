//
// Created by saleh on 1/22/22.
//

#include "CClassifier.h"
#include <cmath>

/**
 * Constructor, it does not do any preparation, please use Prepare().
 * @param dataDir Path to the dump directory of the repository with the trailing `/`
 * @param batchSize Batch-size of the inference run
 * @param dumpTensors true: dump tensors, false: dont dump any tensors
 */
CClassifier::CClassifier(string &dataDir, bool dumpTensors) {
  m_strDataDir = dataDir + "/";
  m_bDumpTensors = dumpTensors;
}

/**
 * Pre-loads the weights.
 * @return 0 on success
 */
int CClassifier::PreloadWeights() {
  try {
    vector<string> vecPath;
    vecPath.push_back("0.conv2d_1.weight.npy");
    vecPath.push_back("1.conv2d_2.weight.npy");
    vecPath.push_back("3.conv2d_3.weight.npy");
    vecPath.push_back("4.conv2d_4.weight.npy");
    vecPath.push_back("7.dense_1.weight.npy");
    vecPath.push_back("8.dense_2.weight.npy");

    for (string &fName: vecPath) {
      std::string npyPath = m_strDataDir + "weights/numpy_npy/" + fName;
      m_vNumpyBuff.push_back(cnpy::npy_load(npyPath));
      std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());
      m_vWeights.push_back(
          CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          )
      );
    }

    return 0;
  }catch(exception &e){
    return 1;
  }
}

/**
 * Pre-loads the biases.
 * @return 0 on success
 */
int CClassifier::PreloadBiases() {
  try{
    vector<string> vecPath;
    vecPath.push_back("0.conv2d_1.bias.npy");
    vecPath.push_back("1.conv2d_2.bias.npy");
    vecPath.push_back("3.conv2d_3.bias.npy");
    vecPath.push_back("4.conv2d_4.bias.npy");
    vecPath.push_back("7.dense_1.bias.npy");
    vecPath.push_back("8.dense_2.bias.npy");

    for(string &fName: vecPath){
      std::string npyPath = m_strDataDir + "weights/numpy_npy/" + fName;
      m_vNumpyBuff.push_back(cnpy::npy_load(npyPath));
      std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());
      m_vBiases.push_back(
          CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          )
      );
    }

    return 0;
  }catch(exception &e){
    return 1;
  }
}

/**
 * Pre-loads the dataset along with the confirmation tensor.
 * @return 0 on success
 */
int CClassifier::PreloadTestSet() {
  try{
    vector<string> vecPath;
    vecPath.push_back("input_test_data.npy");
    vecPath.push_back("input_test_label.npy");
    vecPath.push_back("confirmation.npy");

    int i=0;

    for(string &fName: vecPath){
      std::string npyPath = m_strDataDir + fName;
      m_vNumpyBuff.push_back(cnpy::npy_load(npyPath));
      std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());
      if(i==0) {
        m_oTestSetData = CTensorPtr<float>(
            new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
        );
      }else{
        if(i==1){
          m_oTestSetLabel = CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          );
        }else{
          m_oConfirmation = CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          );
        }
      }
      i++;
    }

    return 0;
  }catch(exception &e){
    return 1;
  }
}

/**
 * Pre-loads the weights, the biases, the confirmation tensor, and the dataset.
 * @return 0 on success
 */
int CClassifier::PreloadAll() {
  int retVal = 0;
  retVal += PreloadWeights();
  retVal += PreloadBiases();
  retVal += PreloadTestSet();
  return retVal;
}

/**
 * Checks the content of the loaded tensor with a known pattern to make sure
 * that everything is working as they should.
 * @return true on success
 */
bool CClassifier::CheckValidityOfNumpyData() {
  auto shape = m_oConfirmation->GetShape();
  float *ptrBuff = m_oConfirmation->Get();
  if(shape.size()!=1) return false;
  if(shape[0]!=1234) return false;
  for(unsigned i=0; i<shape[0]; i++){
    if((float)i != ptrBuff[i]){
      return false;
    }
  }
  return true;
}

/**
 * Prepares the classifier (loads the necessary weights, biases, and dataset;
 *                                          along with doing some sanity checks)
 * @return returns 0 on success
 */
int CClassifier::Prepare() {
  if(PreloadAll()!=0) return 1;
  if(!CheckValidityOfNumpyData()) return 2;
  return 0;
}

/**
 * Performs Conv2D operation with valid or same padding, fixed stride of 1, and in NHWC format.
 * This kernel implements the trailing activation (forced relu).
 *
 * @param inputTn The input tensor, row-major.
 * @param weightTn The weight tensor.
 * @param biasTn The bias tensor, row-major.
 * @param isValidPadding True: valid-padding, False: same-padding
 * @return resulted tensor
 */
CTensorPtr<float> CClassifier::LayerConv2D(CTensorPtr<float> inputTn,
                                            CTensorPtr<float> weightTn,
                                            CTensorPtr<float> biasTn,
                                            bool isValidPadding) {
  CTensorPtr<float> outputTn;
  const auto shapeI = inputTn->GetShape(); // N,H,W,Cin
  const auto shapeW = weightTn->GetShape(); // R,S,Cin,Cout
  const auto shapeB = biasTn->GetShape(); // Cout

  ConditionCheck(biasTn->GetRank()==1,"The bias tensor should be of rank 1.");
  ConditionCheck(shapeB[0] == shapeW[3],"The shapes of the bias tensor and the weight tensor must match at the last axis.");
  ConditionCheck(inputTn->GetRank()==4, "The input tensor with batch size of 1 must be expanded dim at axis 0.")
  ConditionCheck(weightTn->GetRank()==4, "The weight tensor with size of 1 in any of its axes must be expanded dim at the respective axes.")

  unsigned B, H, W, Hout, Wout, Cin, Cout, R, S, padLenH, padLenW;

  B = shapeI[0];
  Cin = shapeI[3];
  Cout = shapeW[3];
  H = shapeI[1];
  W = shapeI[2];
  R = shapeW[0]; // for H
  S = shapeW[1]; // for W

  if(isValidPadding){
    // Forced stride of 1 for i and j.
    Hout = shapeI[1] - shapeW[0] + 1;
    Wout = shapeI[2] - shapeW[1] + 1;
    outputTn = CTensorPtr<float>(new CTensor<float>({shapeI[0], Hout, Wout, shapeW[3]}));
    padLenH = padLenW = 0;
  }else{
    // Forced stride of 1 for i and j.
    Hout = shapeI[1];
    Wout = shapeI[2];
    outputTn = CTensorPtr<float>(new CTensor<float>({shapeI[0], Hout, Wout, shapeW[3]}));
    padLenH = (R - 1);
    padLenW = (S - 1);
  }

  const auto tnI = inputTn->Get();
  const auto tnW = weightTn->Get();
  const auto tnB = biasTn->Get();
  const auto tnO = outputTn->Get();

  if(isValidPadding){

    for (unsigned b=0; b<B; b++) { // Batch
      for (unsigned o=0; o<Cout; o++) { // Cout
        for (unsigned h = 0; h < H; h++) { // H
          for (unsigned w = 0; w < W; w++) { // W

            float sum = 0;
            // ---------------------
            if((h + R -1 < H) && (w + S -1 < W)) {
              for (unsigned c = 0; c < Cin; c++) { // Cin
                for (unsigned j = 0; (j < R); j++) { // R
                  for (unsigned i = 0; (i < S); i++) { // S
                    sum +=
                        tnI[b * H * W * Cin + (h + j) * W * Cin + (w + i) * Cin + c] *
                            tnW[(j) * S * Cin * Cout + (i) * Cin * Cout + c * Cout + o];
                  }
                }
              }

              // ---------------------
              float activated = sum + tnB[o];
              if (activated < 0) activated = 0;
              tnO[b * Hout * Wout * Cout + h * Wout * Cout + w * Cout + o] = activated;
            }
            // ---------------------
          }
        }
      }
    }

  }else{
    int padStartH, padEndH, padStartW, padEndW, boundStartH, boundEndH, boundStartW, boundEndW;

    padStartH = (int)floor((float)padLenH/2.0f);
    padEndH = (int)ceil((float)padLenH/2.0f);
    padStartW = (int)floor((float)padLenW/2.0f);
    padEndW = (int)ceil((float)padLenW/2.0f);

    boundStartH = -1 * padStartH;
    boundEndH = H - (R-(padEndH+1));

    boundStartW = -1 * padStartW;
    boundEndW = W - (S-(padEndW+1));

    for (int b=0; b<B; b++) { // Batch
      for (int o=0; o<Cout; o++) { // Cout
        for (int h = boundStartH; h < boundEndH; h++) { // H
          for (int w = boundStartW; w < boundEndW; w++) { // W

            float sum = 0;
            // ---------------------
            for (int c = 0; c < Cin; c++) { // Cin
              for (unsigned j = 0; (j < R); j++) { // R
                for (unsigned i = 0; (i < S); i++) { // S

                  size_t indexI =
                      b * H * W * Cin +
                      (h + j) * W * Cin +
                      (w + i) * Cin +
                      c;

                  size_t indexW =
                      (j) * S * Cin * Cout +
                      (i) * Cin * Cout +
                      c * Cout +
                      o;

                  float valI;
                  if((h+j>=0 && h+j<H)&&(w+i>=0 && w+i<W)){
                    valI = tnI[indexI];
                  }else{
                    valI = 0;
                  }

                  sum += valI * tnW[indexW];
                }
              }
            }
            // ---------------------
            size_t indexO = b*Hout*Wout*Cout + (h+padStartH)*Wout*Cout + (w+padStartW)*Cout + o;
            float activated = sum + tnB[o];
            if(activated<0) activated = 0;
            tnO[indexO] = activated;
          }
        }
      }
    }

  }

  return outputTn;
}

void CClassifier::DumpTensor(CTensorPtr<float> inputTn, string nameTag) {
  if(m_bDumpTensors){
    string path = globalArgDataPath + "/inference_outputs_cpp/";
    DumpToNumpyFile<float>(nameTag, inputTn, path);
  }
}

void CClassifier::DumpTensor(CTensorPtr<unsigned> inputTn, string nameTag) {
  if(m_bDumpTensors) {
    string path = globalArgDataPath + "/inference_outputs_cpp/";
    DumpToNumpyFile<unsigned>(nameTag, inputTn, path);
  }
}

void CClassifier::DumpTensor(CTensorPtr<int> inputTn, string nameTag) {
  if(m_bDumpTensors) {
    string path = globalArgDataPath + "/inference_outputs_cpp/";
    DumpToNumpyFile<int>(nameTag, inputTn, path);
  }
}

void CClassifier::Inference() {

  auto conv1 = LayerConv2D(m_oTestSetData, m_vWeights[0], m_vBiases[0], false);
  DumpTensor(conv1, "0.output.tensor.npy");

  auto conv2 = LayerConv2D(conv1, m_vWeights[1], m_vBiases[1], true);
  DumpTensor(conv2, "1.output.tensor.npy");

  auto pool1 = LayerMaxPool2D(conv2, {2, 2});
  DumpTensor(pool1, "2.output.tensor.npy");

  auto conv3 = LayerConv2D(pool1, m_vWeights[2], m_vBiases[2], false);
  DumpTensor(conv3, "3.output.tensor.npy");

  auto conv4 = LayerConv2D(conv3, m_vWeights[3], m_vBiases[3], true);
  DumpTensor(conv4, "4.output.tensor.npy");

  auto pool2 = LayerMaxPool2D(conv4, {2, 2});
  DumpTensor(pool2, "5.output.tensor.npy");

  auto flatten = LayerFlatten(pool2);
  DumpTensor(flatten, "6.output.tensor.npy");

  auto dense1 = LayerDense(flatten, m_vWeights[4], m_vBiases[4]);
  auto denseRelu1 = LayerReLU(dense1);
  DumpTensor(denseRelu1, "7.output.tensor.npy");

  auto dense2 = LayerDense(denseRelu1, m_vWeights[5], m_vBiases[5]);
  auto denseSoftMax2 = LayerSoftMax(dense2);
  DumpTensor(denseSoftMax2, "8.output.tensor.npy");
}

/**
 * 2D MaxPooling with forced `valid` padding.
 * @param inputTn The input tensor in `BHWC` format
 * @param poolSize The pool size in `HW` format, for example {2,2}
 * @return The output tensor
 */
CTensorPtr<float> CClassifier::LayerMaxPool2D(CTensorPtr<float> inputTn, const vector<unsigned> &poolSize) {
  CTensorPtr<float> outputTn;
  const auto shapeI = inputTn->GetShape(); // N,H,W,Cin

  ConditionCheck(inputTn->GetRank()==4, "The input tensor with batch size of 1 must be expanded dim at axis 0.")
  ConditionCheck(poolSize.size()==2, "The pool size for 2D max pooling must have two elements.")

  unsigned B, H, W, Cin, R, S, Hout, Wout;

  B = shapeI[0];
  H = shapeI[1];
  W = shapeI[2];
  Cin = shapeI[3];
  R = poolSize[0];
  S = poolSize[1];

  Hout = ceil(((float)(H-R+1))/((float)R)); // stride = R
  Wout = ceil(((float)(W-S+1))/((float)S)); // stride = S

  outputTn = CTensorPtr<float>(new CTensor<float>({B, Hout, Wout, Cin}));

  const float* tnI = inputTn->Get();
  float* tnO = outputTn->Get();

  for(unsigned b=0; b<B; b++){
    for(unsigned c=0; c<Cin; c++){
      for(unsigned h=0; h<H; h+=R){
        for(unsigned w=0; w<W; w+=S){

          if( (h+R>=0 && h+R-1<H) && (w+S>=0 && w+S-1<W) ){
            float maxVal = -1 * std::numeric_limits<float>::infinity();

            for(unsigned j=0; j<R; j++){
              for(unsigned i=0; i<S; i++){
                size_t indexI =
                        b*H*W*Cin+
                        (h+j)*W*Cin+
                        (w+i)*Cin+
                        c;
                if(tnI[indexI]>maxVal) {
                  maxVal = tnI[indexI];
                }
              }
            }

            size_t indexO =
                  b*Hout*Wout*Cin+
                  (h/R)*Wout*Cin+
                  (w/S)*Cin+
                  c;
            tnO[indexO] = maxVal;
          }

        }
      }
    }
  }

  return outputTn;
}

/**
 * Implements `Flatten` funtionality, keeps the batch axis in tact.
 * @param inputTn The input tensor
 * @return The output tensor
 */
CTensorPtr<float> CClassifier::LayerFlatten(CTensorPtr<float> inputTn) {
  ConditionCheck(inputTn->GetRank()>=2, "The rank of the input tensor should be >=2.")
  const auto oldShape = inputTn->GetShape();
  unsigned flattenShape = std::accumulate(
      std::begin(oldShape)+1, // we only want to flatten the `HWC` axes, not the `B` axis
      std::end(oldShape),
      1,
      std::multiplies<unsigned>()
  );

  CTensorPtr<float> outputTn =
      CTensorPtr<float>(new CTensor<float>({oldShape[0], flattenShape}, inputTn->Get()));
  return outputTn;
}

/**
 * Implements the `Dense` layer without the activation functionality.
 * @param inputTn The input tensor
 * @param weightTn The weight tensor
 * @param biasTn The bias tensor
 * @return The output tensor
 */
CTensorPtr<float> CClassifier::LayerDense(CTensorPtr<float> inputTn,
                                          CTensorPtr<float> weightTn,
                                          CTensorPtr<float> biasTn) {
  auto shape1 = inputTn->GetShape();
  auto shape2 = weightTn->GetShape();
  auto matrixH1  = shape1[0];
  auto matrixW1  = shape1[1];
  auto matrixH2  = shape2[0];
  auto matrixW2  = shape2[1];
  ConditionCheck(matrixW1==matrixH2, "Unequal inputTn[1] and weightTn[0].");
  CTensorPtr<float> outputTn(new CTensor<float>({matrixH1, matrixW2}));

  size_t indexS1, indexS2, indexD;
  const float *tnI = inputTn->Get();
  const float *tnW = weightTn->Get();
  const float *tnB = biasTn->Get();
  float *tnO = outputTn->Get();

  // for element of output of matrixH1 x matrixW2
  for(unsigned j=0;j<matrixH1;j++){
    for(unsigned i=0;i<matrixW2;i++){
      //mat1: select row j
      //mat2: select col i
      float sum=0;
      for(unsigned mat1_x=0;mat1_x<matrixW1;mat1_x++)
      {
        indexS1 = j*matrixW1 + mat1_x;
        indexS2 = mat1_x*matrixW2 + i;
        sum += tnI[indexS1] * tnW[indexS2];
      }
      // for element of output of matrixH1 x matrixW2
      indexD = j*matrixW2 + i;
      tnO[indexD] = sum + tnB[i];
    }
  }

  return outputTn;
}

/**
 * Implements `ReLU`
 * @param inputTn The input tensor of rank 1 and greater.
 * @return The output tensor.
 */
CTensorPtr<float> CClassifier::LayerReLU(CTensorPtr<float> inputTn) {
  ConditionCheck(inputTn->GetLen()!=0, "The input tensor is of length zero!");
  CTensorPtr<float> outputTn(new CTensor<float>(inputTn->GetShape()));
  const size_t len = inputTn->GetLen();
  const float *tnI = inputTn->Get();
  float *tnO = outputTn->Get();
  for(size_t i=0;i<len;i++){
    tnO[i] = (tnI[i]>0) ? tnI[i] : 0;
  }
  return outputTn;
}

/**
 * Implements `SoftMax`
 * @param inputTn The input tensor of rank 2 in BxClassCount format.
 * @return The output tensor BxClassCount
 */
CTensorPtr<float> CClassifier::LayerSoftMax(CTensorPtr<float> inputTn) {
  ConditionCheck(inputTn->GetRank()==2, "The input tensor must be of rank 2. The first axis is for the batch.")

  CTensorPtr<float> outputTn(new CTensor<float>(inputTn->GetShape()));

  const auto shapeI = inputTn->GetShape();
  const float* tnI = inputTn->Get();
  float* tnO = outputTn->Get();

  unsigned B, C;
  B = shapeI[0];
  C = shapeI[1];

  for(unsigned b=0; b<B; b++){
    float maxVal = -1 * std::numeric_limits<float>::infinity();
    for(unsigned c=0; c<C; c++){
      if(tnI[b*C+c] > maxVal){
        maxVal = tnI[b*C+c];
      }
    }
    float sum = 0;
    for(unsigned c=0; c<C; c++){
      sum += exp(tnI[b*C+c] - maxVal);
    }

    for(unsigned c=0; c<C; c++){
      tnO[b*C+c] = exp(tnI[b*C+c]-maxVal) / sum;
    }
  }

  return outputTn;
}



