///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#ifndef DEM_BONES_DEM_BONES
#define DEM_BONES_DEM_BONES

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <algorithm>
#include <queue>
#include <vector>
#include <set>
#include <stack>
#include <iostream>
#include "ConvexLS.h"


#ifndef DEM_BONES_MAT_BLOCKS
#include "MatBlocks.h"
#define DEM_BONES_DEM_BONES_MAT_BLOCKS_UNDEFINED
#endif


namespace Dem
{

/** @mainpage Overview
	Main elements:
	- @ref DemBones : base class with the core solver using relative bone transformations DemBones::m
	- @ref DemBonesExt : extended class to handle hierarchical skeleton with local rotations/translations and bind matrices
	- DemBones/MatBlocks.h: macros to access sub-blocks of packing transformation/position matrices for convenience

	Include DemBones/DemBonesExt.h (or DemBones/DemBones.h) with optional DemBones/MatBlocks.h then follow these steps to use the library:
	-# Load required data in the base class:
		- Rest shapes: DemBones::u, DemBones::fv, DemBones::nV
		- Sequence: DemBones::v, DemBones::nF, DemBones::fStart, DemBones::subjectID, DemBones::nS
		- Number of bones DemBones::nB
	-# Load optional data in the base class:
		- Skinning weights DemBones::w and weights soft-lock DemBones::lockW
		- Bone transformations DemBones::m and bones hard-lock DemBones::lockM
	-# [@c optional] Set parameters in the base class: 
		- DemBones::nIters
		- DemBones::nInitIters
		- DemBones::nTransIters, DemBones::transAffine, DemBones::transAffineNorm
		- DemBones::nWeightsIters, DemBones::nnz, DemBones::weightsSmooth, DemBones::weightsSmoothStep, DemBones::weightEps
	-# [@c optional] Setup extended class:
		- Load data: DemBonesExt::parent, DemBonesExt::preMulInv, DemBonesExt::rotOrder, DemBonesExt::orient, DemBonesExt::bind
		- Set parameter DemBonesExt::bindUpdate
	-# [@c optional] Override callback functions (cb...) in the base class @ref DemBones
	-# Call decomposition function DemBones::compute(), DemBones::computeWeights(), DemBones::computeTranformations(), or DemBones::init()
	-# [@c optional] Get local transformations/bind poses with DemBonesExt::computeRTB() 
*/

/** @class DemBones DemBones.h "DemBones/DemBones.h"
	@brief Smooth skinning decomposition with rigid bones and sparse, convex weights
	
	@details Setup the required data, parameters, and call either compute(), computeWeights(), computeTranformations(), or init().
	
	Callback functions and read-only values can be used to report progress and stop on convergence: cbInitSplitBegin(), cbInitSplitEnd(), 
	cbIterBegin(), cbIterEnd(), cbWeightsBegin(), cbWeightsEnd(), cbTranformationsBegin(), cbTransformationsEnd(), cbTransformationsIterBegin(),
	cbTransformationsIterEnd(), cbWeightsIterBegin(), cbWeightsIterEnd(), rmse(), #iter, #iterTransformations, #iterWeights.

	@b _Scalar is the floating-point data type. @b _AniMeshScalar is the floating-point data type of mesh sequence #v.
*/
template<class _Scalar, class _AniMeshScalar>
class DemBones {
public: 
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	using MatrixX=Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Matrix4=Eigen::Matrix<_Scalar, 4, 4>;
	using Matrix3=Eigen::Matrix<_Scalar, 3, 3>;
	using VectorX=Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;
	using Vector4=Eigen::Matrix<_Scalar, 4, 1>;
	using Vector3=Eigen::Matrix<_Scalar, 3, 1>;
	using SparseMatrix=Eigen::SparseMatrix<_Scalar>;
	using Triplet=Eigen::Triplet<_Scalar>;

	//! [@c parameter] Number of global iterations, @c default = 30
	int nIters;

	//! [@c parameter] Number of clustering update iterations in the initalization, @c default = 10
	int nInitIters;

	//! [@c parameter] Number of bone transformations update iterations per global iteration, @c default = 5
	int nTransIters;
	//! [@c parameter] Translations affinity soft constraint, @c default = 10.0
	_Scalar transAffine;
	//! [@c parameter] p-norm for bone translations affinity soft constraint, @c default = 4.0
	_Scalar transAffineNorm;
	
	//! [@c parameter] Number of weights update iterations per global iteration, @c default = 3
	int nWeightsIters;
	//! [@c parameter] Number of non-zero weights per vertex, @c default = 8
	int nnz;
	//! [@c parameter] Weights smoothness soft constraint, @c default = 1e-4
	_Scalar weightsSmooth;	
	//! [@c parameter] Step size for the weights smoothness soft constraint, @c default = 1.0
	_Scalar weightsSmoothStep;
	//! [@c parameter] Epsilon for weights solver, @c default = 1e-15
	_Scalar weightEps;
	//! Minimum split/prune cluster size. Values <= 0 use nV/500.
	int clusterThreshold;
	//! 0 uses graph propagation, 1 uses global best-error assignment.
	int labelAssignmentMode;
	//! Number of same-parent spatial neighbors inserted with each split seed.
	int splitSeedNeighbors;
	//! Number of initialization refinement steps completed before pruning.
	int pruneDelay;
	//! Additional consecutive zero-growth split rounds allowed.
	int zeroGrowthPatience;
	//! 0 disables point-cloud weight pre-smoothing, 1 uses the KNN solver.
	int pointCloudSmoothMode;
	
	/** @brief Constructor and setting default parameters
	*/
	DemBones():	nIters(30), nInitIters(10),
			nTransIters(5),	transAffine(_Scalar(10)), transAffineNorm(_Scalar(4)),
			nWeightsIters(3), nnz(8), weightsSmooth(_Scalar(1e-4)), weightsSmoothStep(_Scalar(1)),
			weightEps(_Scalar(1e-15)), clusterThreshold(3),
			labelAssignmentMode(1), splitSeedNeighbors(0), pruneDelay(0),
			zeroGrowthPatience(0), pointCloudSmoothMode(1),
			knn_graph_computed(false),  // 初始化缓存标志
			iter(_iter), iterTransformations(_iterTransformations), iterWeights(_iterWeights) {
		clear();
	}
	
	//! Number of vertices, typically indexed by @p i
	int nV;
	//! Number of bones, typically indexed by @p j
	int nB;					
	//! Number of subjects, typically indexed by @p s
	int nS;		
	//! Number of total frames, typically indexed by @p k, #nF = #fStart(#nS)
	int nF;	

	//! Start frame indices, @c size = #nS+1, #fStart(@p s), #fStart(@p s+1) are data frames for subject @p s
	Eigen::VectorXi fStart;
	//! Subject index of the frame, @c size = #nF, #subjectID(@p k)=@p s, where #fStart(@p s) <= @p k < #fStart(<tt>s</tt>+1)
	Eigen::VectorXi subjectID;

	//! Geometry at the rest poses, @c size = [3*#nS, #nV], #u.@a col(@p i).@a segment(3*@p s, 3) is the rest pose of vertex @p i of subject @p s
	MatrixX u;

	//! Skinning weights, @c size = [#nB, #nV], #w.@a col(@p i) are the skinning weights of vertex @p i, #w(@p j, @p i) is the influence of bone @p j to vertex @p i
	SparseMatrix w;

	//! Skinning weights lock control, @c size = #nV, #lockW(@p i) is the amount of input skinning weights will be kept for vertex @p i, where 0 (no lock) <= #lockW(@p i) <= 1 (full lock)
	VectorX lockW;

	/** @brief Bone transformations, @c size = [4*#nF*4, 4*#nB], #m.@a blk4(@p k, @p j) is the 4*4 relative transformation matrix of bone @p j at frame @p k
		@details Note that the transformations are relative, that is #m.@a blk4(@p k, @p j) brings the global transformation of bone @p j from the rest pose to the pose at frame @p k.
	*/
	MatrixX m;

	//! Bone transformation lock control, @c size = #nB, #lockM(@p j) is the amount of input transformations will be kept for bone @p j, where #lockM(@p j) = 0 (no lock) or 1 (lock)
	Eigen::VectorXi lockM;

	//! Animated mesh sequence, @c size = [3*#nF, #nV], #v.@a col(@p i).@a segment(3*@p k, 3) is the position of vertex @p i at frame @p k
	Eigen::Matrix<_AniMeshScalar, Eigen::Dynamic, Eigen::Dynamic> v;
	
	//! Mesh topology, @c size=[<tt>number of polygons</tt>], #fv[@p p] is the vector of vertex indices of polygon @p p
	std::vector<std::vector<int>> fv;

	//! [<tt>zero indexed</tt>, <tt>read only</tt>] Current global iteration number that can be used for callback functions
	const int& iter;

	//! [<tt>zero indexed</tt>, <tt>read only</tt>] Current bone transformations update iteration number that can be used for callback functions
	const int& iterTransformations;

	//! [<tt>zero indexed</tt>, <tt>read only</tt>] Current weights update iteration number that can be used for callback functions
	const int& iterWeights;
	

	// Global update keep bones (bones not removed yet)
	Eigen::VectorXi keep_bones;	


	MatrixX ErrVtxBoneAll;


	/** @brief Clear all data
	*/
	void clear() {
		nV=nB=nS=nF=0;
		fStart.resize(0);
		subjectID.resize(0);
		u.resize(0, 0);
		w.resize(0, 0);
		lockW.resize(0);
		m.resize(0, 0);
		lockM.resize(0);
		v.resize(0, 0);
		fv.resize(0);
		modelSize=-1;
		laplacian.resize(0, 0);
		
		// 清除KNN缓存
		cached_knn_graph.clear();
		knn_graph_computed = false;
	}

	/** @brief Initialize missing skinning weights and/or bone transformations
		@details Depending on the status of #w and #m, this function will:
			- Both #w and #m are already set: do nothing
			- Only one in #w or #m is missing (zero size): initialize missing matrix, i.e. #w (or #m)
			- Both #w and #m are missing (zero size): initialize both with rigid skinning using approximately #nB bones, i.e. values of #w are 0 or 1.
			LBG-VQ clustering is peformed using mesh sequence #v, rest pose geometries #u and topology #fv.
			@b Note: as the initialization does not use exactly #nB bones, the value of #nB could be changed when both #w and #m are missing.
			
		This function is called at the begining of every compute update functions as a safeguard.
	*/
	void init() {
		if (modelSize<0) modelSize=sqrt((u-(u.rowwise().sum()/nV).replicate(1, nV)).squaredNorm()/nV/nS);
		if (fv.empty() || fv.size() == 0) {
			if ((labelAssignmentMode == 0 || splitSeedNeighbors > 0 || pointCloudSmoothMode != 0)
				&& knn_laplacian.cols() != nV) computeKNNSmoothSolver(8);
		} else {
			if (laplacian.cols() != nV) computeSmoothSolver();
		}

		if (((int)w.rows()!=nB)||((int)w.cols()!=nV)) { //No skinning weight
			if (((int)m.rows()!=nF*4)||((int)m.cols()!=nB*4)) { //No transformation
				int targetNB=nB;
				//LBG-VQ

				nB=1;
				label=Eigen::VectorXi::Zero(nV);
				computeTransFromLabel();
		
				// std::cout << "Computed computeTransFromLabel from single label" << std::endl;

				bool cont=true;
				int stalledRounds=0;
				while (cont) {
					cbInitSplitBegin();
					int prev=nB;
					int threshold=clusterThreshold>0 ? clusterThreshold : std::max(1, int(nV/500));
					split(targetNB, threshold);
					for (int rep=0; rep<nInitIters; rep++) {
						computeTransFromLabel();
						reassignLabels();
						if (rep>=pruneDelay) pruneBones(threshold);
					}
					if (nB>prev) stalledRounds=0; else stalledRounds++;
					cont=(nB<targetNB)&&((nB>prev)||(stalledRounds<=zeroGrowthPatience));
					if (nB==prev) std::cout << "nB == prev!!!" << std::endl;
					cbInitSplitEnd();
				}
				lockM=Eigen::VectorXi::Zero(nB);
				labelToWeights();
			} else initWeights(); //Has transformations
		} else { //Has skinning weights
			if (((int)m.rows()!=nF*4)||((int)m.cols()!=nB*4)) { //No transformation
				m=Matrix4::Identity().replicate(nF, nB);
				lockM=Eigen::VectorXi::Zero(nB);
			}
		}

		if (lockW.size()!=nV) lockW=VectorX::Zero(nV);
		if (lockM.size()!=nB) lockM=Eigen::VectorXi::Zero(nB);
		
		// Set keep bones to all bones 
		keep_bones = Eigen::ArrayXi::LinSpaced(nB, 0, nB-1);
	}

	/** @brief Update bone transformations by running #nTransIters iterations with #transAffine and #transAffineNorm regularizers
		@details Required input data:
			- Rest shapes: #u, #fv, #nV
			- Sequence: #v, #nF, #fStart, #subjectID, #nS
			- Number of bones: #nB

		Optional input data:
			- Skinning weights: #w, #lockW
			- Bone transformations: #m, #lockM

		Output: #m. Missing #w and/or #m (with zero size) will be initialized by init().
	*/
	void computeTranformations() {
		if (nTransIters==0) return;

		init();
		cbTranformationsBegin();

		compute_vuT();
		compute_uuT();

		for (_iterTransformations=0; _iterTransformations<nTransIters; _iterTransformations++) {
			cbTransformationsIterBegin();
			#pragma omp parallel for
			for (int k=0; k<nF; k++)
				for (int j=0; j<nB; j++) 
					if (lockM(j)==0) {
						Matrix4 qpT=vuT.blk4(k, j);
						for (int it=uuT.outerIdx(j); it<uuT.outerIdx(j+1); it++)
							if (uuT.innerIdx(it)!=j) qpT-=m.blk4(k, uuT.innerIdx(it))*uuT.val.blk4(subjectID(k), it);
						qpT2m(qpT, k, j);
					}
			if (cbTransformationsIterEnd()) return;
		}
		
		cbTransformationsEnd();
	}

	/** @brief Update skinning weights by running #nWeightsIters iterations with #weightsSmooth and #weightsSmoothStep regularizers
		@details Required input data:
			- Rest shapes: #u, #fv, #nV
			- Sequence: #v, #nF, #fStart, #subjectID, #nS
			- Number of bones: #nB

		Optional input data:
			- Skinning weights: #w, #lockW
			- Bone transformations: #m, #lockM

		Output: #w. Missing #w and/or #m (with zero size) will be initialized by init().

	*/
	void compute_errorVtxBoneALL(){
		ErrVtxBoneAll.resize(nV,nB);
		ErrVtxBoneAll.setZero();
		for (int i = 0; i < nV; ++i)
			for (int j = 0; j < nB; ++j)
				ErrVtxBoneAll(i,j) = errorVtxBone(i,j);
	}

	void computeWeights() {
		if (nWeightsIters==0) return;
		
		init();
		cbWeightsBegin();
		
		compute_mTm();

		aTb=MatrixX::Zero(nB, nV);
		wSolver.init(nnz);
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> trip;
		trip.reserve(nV*nnz);

		for (_iterWeights=0; _iterWeights<nWeightsIters; _iterWeights++) {
			cbWeightsIterBegin();

			compute_ws();
			compute_aTb();

			double reg_scale=pow(modelSize, 2)*nF;

			trip.clear();
			#pragma omp parallel for
			for (int i=0; i<nV; i++) {
				MatrixX aTai;
				compute_aTa(i, aTai);
				aTai=(1-lockW(i))*(aTai/reg_scale+weightsSmooth*MatrixX::Identity(nB, nB))+lockW(i)*MatrixX::Identity(nB, nB);
				VectorX aTbi=(1-lockW(i))*(aTb.col(i)/reg_scale+weightsSmooth*ws.col(i))+lockW(i)*w.col(i);

				VectorX x=(1-lockW(i))*ws.col(i)+lockW(i)*w.col(i);
				Eigen::ArrayXi idx=Eigen::ArrayXi::LinSpaced(nB, 0, nB-1);
				std::sort(idx.data(), idx.data()+nB, [&x](int i1, int i2) { return x(i1)>x(i2); });
				int nnzi=std::min(nnz, nB);
				while (x(idx(nnzi-1))<weightEps) nnzi--;

				VectorX x0=w.col(i).toDense().cwiseMax(0.0);
				x=indexing_vector(x0, idx.head(nnzi));
				_Scalar s=x.sum();
				if (s>_Scalar(0.1)) x/=s; else x=VectorX::Constant(nnzi, _Scalar(1)/nnzi);

				wSolver.solve(indexing_row_col(aTai, idx.head(nnzi), idx.head(nnzi)), 
							  indexing_vector(aTbi, idx.head(nnzi)), x, true, true);
				

				#pragma omp critical
				for (int j=0; j<nnzi; j++)
					if (x(j)!=0) trip.push_back(Triplet(idx[j], i, x(j)));
				
			}

			w.resize(nB, nV);
			w.setFromTriplets(trip.begin(), trip.end());
			
			if (cbWeightsIterEnd()) return;
		}
		
		cbWeightsEnd();
	}

	/** @brief Skinning decomposition by #nIters iterations of alternative updating weights and bone transformations
		@details Required input data:
			- Rest shapes: #u, #fv, #nV
			- Sequence: #v, #nF, #fStart, #subjectID, #nS
			- Number of bones: #nB

		Optional input data:
			- Skinning weights: #w
			- Bone transformations: #m

		Output: #w, #m. Missing #w and/or #m (with zero size) will be initialized by init().
	*/
	void compute() {
		printf("Computing Init Bones\n");
		init();

		for (_iter=0; _iter<nIters; _iter++) {
			cbIterBegin();
			computeTranformations();
			computeWeights();
			if (cbIterEnd()) break;
		}
	}

	//! @return Root mean squared reconstruction error
	_Scalar rmse() {
		_Scalar e=0;
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			_Scalar ei=0;
			Matrix4 mki;
			for (int k=0; k<nF; k++) {
				mki.setZero();
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) mki+=it.value()*m.blk4(k, it.row());
				ei+=(mki.template topLeftCorner<3, 3>()*u.vec3(subjectID(k), i)+mki.template topRightCorner<3, 1>()-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
			}
			#pragma omp atomic
			e+=ei;
		}
		return std::sqrt(e/nF/nV);
	}


	std::vector<_Scalar> vertex_rmse(std::vector<int> vert_inds) {
		std::vector<_Scalar> vert_recon_err_list;
		vert_recon_err_list.resize(vert_inds.size());

		for (int ind=0; ind<vert_inds.size(); ind++) {
			int i = vert_inds[ind];
			_Scalar ei=0;
			#pragma omp parallel for
			for (int k=0; k<nF; k++) {
				Matrix4 mki;
				mki.setZero();	
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) mki+=it.value()*m.blk4(k, it.row());
				#pragma omp atomic
				ei+=(mki.template topLeftCorner<3, 3>()*u.vec3(subjectID(k), i)+mki.template topRightCorner<3, 1>()-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
			}
			vert_recon_err_list[ind] = std::sqrt(ei/nF);
		}
		return vert_recon_err_list;
	}

	std::vector<_Scalar> vertex_max_rmse(std::vector<int> vert_inds) {
		std::vector<_Scalar> vert_recon_err_list;
		vert_recon_err_list.resize(vert_inds.size());

		#pragma omp parallel for
		for (int ind=0; ind<vert_inds.size(); ind++) {
			int i = vert_inds[ind];
			_Scalar ei=0;
			for (int k=0; k<nF; k++) {
				Matrix4 mki;
				mki.setZero();	
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) mki+=it.value()*m.blk4(k, it.row());
				_Scalar eif=(mki.template topLeftCorner<3, 3>()*u.vec3(subjectID(k), i)+mki.template topRightCorner<3, 1>()-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();			

				ei= eif > ei ? eif:ei;
			}
			#pragma omp critical
			vert_recon_err_list[ind] = ei;
		}
		return vert_recon_err_list;
	}


	/**
	 * 计算顶点重建位置
	 * 
	 * @param vert_inds 需要重建的顶点索引列表
	 * @return MatrixX 重建后的顶点位置矩阵，形状为(3*nF, n_verts)
	 *         - 每3行代表一帧中所有顶点的位置
	 *         - 每列代表一个顶点在所有帧中的位置
	 */
	MatrixX compute_reconstruction(std::vector<int> vert_inds) {
		// 初始化结果矩阵
		MatrixX reconstructed_pose;
		// 设置矩阵维度：3*nF行(每帧3个坐标)，n_verts列(每个顶点)
		reconstructed_pose.resize(3*nF, int(vert_inds.size()));
		// 将矩阵所有元素初始化为0
		reconstructed_pose.setZero();

		// 如果没有需要重建的顶点，直接返回空矩阵
		if(vert_inds.size() == 0)
			return reconstructed_pose;

		// 使用OpenMP并行计算，对每个顶点独立处理
		#pragma omp parallel for
		for (int ind = 0; ind < int(vert_inds.size()); ++ind) {
			// 获取当前处理的顶点索引
			int i = vert_inds[ind];
			// 用于存储累积变换矩阵
			Matrix4 mki;
			
			// 对每一帧进行处理
			for(int k=0; k<nF; k++) {
				// 重置变换矩阵
				mki.setZero();
				
				// 遍历该顶点的所有骨骼权重
				// w是权重矩阵，i是顶点索引
				// it.row()是骨骼索引，it.value()是权重值
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) {
					// 累加每个骨骼的加权变换矩阵
					// m.blk4(k, it.row())获取第k帧第it.row()个骨骼的变换矩阵
					mki += it.value() * m.blk4(k, it.row());
				}

				// 使用临界区保护共享数据的写入
				#pragma omp critical
				{
					// 计算变换后的顶点位置
					// mki.template topLeftCorner<3, 3>() 获取变换矩阵的旋转部分
					// u.vec3(subjectID(k), i) 获取原始顶点位置
					// mki.template topRightCorner<3, 1>() 获取变换矩阵的平移部分
					reconstructed_pose.vec3(k,ind) = 
						mki.template topLeftCorner<3, 3>() * u.vec3(subjectID(k), i) + 
						mki.template topRightCorner<3, 1>();
				}
			}
		}

		return reconstructed_pose;         
	}

	//! Callback function invoked before each spliting of bone clusters in initialization
	virtual void cbInitSplitBegin() {}
	//! Callback function invoked after each spliting of bone clusters in initialization
	virtual void cbInitSplitEnd() {}

	//! Callback function invoked before each global iteration update
	virtual void cbIterBegin() {}
	//! Callback function invoked after each global iteration update, stop iteration if return true
	virtual bool cbIterEnd() { return false; }

	//! Callback function invoked before each skinning weights update
	virtual void cbWeightsBegin() {}
	//! Callback function invoked after each skinning weights update
	virtual void cbWeightsEnd() {}

	//! Callback function invoked before each bone transformations update
	virtual void cbTranformationsBegin() {}
	//! Callback function invoked after each bone transformations update
	virtual void cbTransformationsEnd() {}

	//! Callback function invoked before each local bone transformations update iteration
	virtual void cbTransformationsIterBegin() {}
	//! Callback function invoked after each local bone transformations update iteration, stop iteration if return true
	virtual bool cbTransformationsIterEnd() { return false; }

	//! Callback function invoked before each local weights update iteration
	virtual void cbWeightsIterBegin() {}
	//! Callback function invoked after each local weights update iteration, stop iteration if return true
	virtual bool cbWeightsIterEnd() { return false; }


	//! mTm.size = (4*nS*nB, 4*nB), where mTm.block<4, 4>(s*nB+i, j) = \sum_{k=fStart(s)}^{fStart(s+1)-1} m.block<3, 4>(k*4, i*4)^T*m.block<3, 4>(k*4, j*4)
	MatrixX mTm;


	//! label(i) is the index of the bone associated with vertex i
	Eigen::VectorXi label;


// private:
	int _iter, _iterTransformations, _iterWeights;

	/** Best rigid transformation from covariance matrix
		@param _qpT is the 4*4 covariance matrix
		@param k is the frame number
		@param j is the bone index
	*/
	void qpT2m(const Matrix4& _qpT, int k, int j) {
		if (_qpT(3, 3)!=0) {
			Matrix4 qpT=_qpT/_qpT(3, 3);
			Eigen::JacobiSVD<Matrix3> svd(qpT.template topLeftCorner<3, 3>()-qpT.template topRightCorner<3, 1>()*qpT.template bottomLeftCorner<1, 3>(), Eigen::ComputeFullU|Eigen::ComputeFullV);
			Matrix3 d=Matrix3::Identity();
			d(2, 2)=(svd.matrixU()*svd.matrixV().transpose()).determinant();
			m.rotMat(k, j)=svd.matrixU()*d*svd.matrixV().transpose();
			m.transVec(k, j)=qpT.template topRightCorner<3, 1>()-m.rotMat(k, j)*qpT.template bottomLeftCorner<1, 3>().transpose();
		}
	}

	/** Fitting error
		@param i is the vertex index
		@param j is the bone index
	*/
	_Scalar errorVtxBone(int i, int j, bool par=true) {
		_Scalar e=0;
		#pragma omp parallel for if(par)
		for (int k=0; k<nF; k++)
			#pragma omp atomic
			e+=(m.rotMat(k, j)*u.vec3(subjectID(k), i)+m.transVec(k, j)-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
		return e;
	}



	// If bones are not defined create them. 
	//! Connected Compononent. Assign each component a different label in the beginning 
	void connected_component(){
		nB=1;
		label=Eigen::VectorXi::Constant(nV, -1);		// Run dfs to find connected components

		// Create adg_matrix of mesh 
		std::vector< std::vector<int> > adj_matrix(nV);

		for (int f = 0; f < (int)fv.size(); ++f){
			int nf = (int)fv[f].size();
			for (int ind1 = 0; ind1 < nf; ++ind1){
				int v1 = fv[f][ind1];
				int v2 = fv[f][(ind1+1)%nf];
				adj_matrix[v1].push_back(v2);
				adj_matrix[v2].push_back(v1);
			}
		}
			
		int current_vertex,current_children, current_label = 0;
		std::stack<int> stack;

		for(int i=0;i<nV;i++){
			if(label(i) != -1) continue; 

			label(i) = current_label;
			stack.push(i);

			while (!stack.empty()){
				current_vertex = stack.top();
				stack.pop();
				for (auto& it: adj_matrix[current_vertex])
					if(label(it)==-1){
						label(it) = current_label;
						stack.push(it);					
					}

			}

			std::cout << "Vertex:" << i << " Label:" << current_label << std::endl;
			current_label++;	

		}

		nB = current_label;	

	}

	void connected_component_simplified(){
    	nB = 1;  // 假设整个网格是一个连通分量
    	label = Eigen::VectorXi::Zero(nV);  // 所有顶点标记为骨骼0
	}	


	//! Comparator for heap with smallest values on top
	struct TripletLess {
		bool operator() (const Triplet& t1, const Triplet& t2) {
			return t1.value()>t2.value();
		}
	};

	void computeLabel_simple() {
		// 直接为每个顶点分配误差最小的骨骼
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			_Scalar minError = std::numeric_limits<_Scalar>::max();
			int bestBone = -1;
			
			for (int j=0; j<nB; j++) {
				_Scalar ej = errorVtxBone(i, j, false);
				if (ej < minError) {
					minError = ej;
					bestBone = j;
				}
			}
			
			label(i) = bestBone;
		}
	}

	void reassignLabels() {
		if (labelAssignmentMode==1) computeLabel_simple(); else computeLabel();
	}


	/** Update labels of vertices
	*/
	void computeLabel() {
		VectorX ei(nV);
		Eigen::VectorXi seed=Eigen::VectorXi::Constant(nB, -1);
		VectorX gMin(nB);
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			int j=label(i);
			if (j!=-1) {
				ei(i)=errorVtxBone(i, j, false);
				if ((seed(j)==-1)||(ei(i)<gMin(j))) {
					#pragma omp critical
					if ((seed(j)==-1)||(ei(i)<gMin(j))) {
						gMin(j)=ei(i);
						seed(j)=i;
					}
				}
			}
		}

		std::priority_queue<Triplet, std::vector<Triplet, Eigen::aligned_allocator<Triplet>>, TripletLess> heap;
		for (int j=0; j<nB; j++) if (seed(j)!=-1) heap.push(Triplet(j, seed(j), ei(seed(j))));

		const bool pointCloud = fv.empty() || fv.size() == 0;
		if (pointCloud) {
			if (knn_laplacian.cols() != nV) computeKNNSmoothSolver(8);
		} else {
			if (laplacian.cols() != nV) computeSmoothSolver();
		}

		std::vector<bool> dirty(nV, true);
		while (!heap.empty()) {
			Triplet top=heap.top();
			heap.pop();
			int i=(int)top.col();
			int j=(int)top.row();
			if (dirty[i]) {
				label(i)=j;
				ei(i)=top.value();
				dirty[i]=false;
				SparseMatrix& adjacency = pointCloud ? knn_laplacian : laplacian;
				for (typename SparseMatrix::InnerIterator it(adjacency, i); it; ++it) {
					int i2=(int)it.row();
					if (dirty[i2]) {
						double tmp=(label(i2)==j)?ei(i2):errorVtxBone(i2, j);
						heap.push(Triplet(j, i2, tmp));
					}
				}
			}
		}

		#pragma omp parallel for
		for (int i=0; i<nV; i++) 
			if (label(i)==-1) {
				_Scalar gMin;
				for (int j=0; j<nB; j++) {
					_Scalar ej=errorVtxBone(i, j, false);
					if ((label(i)==-1)||(gMin>ej)) {
						gMin=ej;
						label(i)=j;
					}
				}
			}
	}

	
	_Scalar rmse_from_cluster(std::vector<int> vert_inds,bool par=true) {
		
		MatrixX cluster_transform=Matrix4::Identity().replicate(nF, 1);
		_Scalar cluster_error = 0;
		#pragma omp parallel for if(par)
		for (int k=0; k<nF; k++) {
			MatrixX qpT=MatrixX::Zero(4, 4);
			for (int ind=0; ind<vert_inds.size(); ind++) {
				int i = vert_inds[ind];
				qpT.blk4(0, 0)+=Vector4(v.vec3(k, i).template cast<_Scalar>().homogeneous())*u.vec3(subjectID(k), i).homogeneous().transpose();
			}

			if (qpT(3, 3)!=0) {
				qpT=qpT/qpT(3, 3);
				Eigen::JacobiSVD<Matrix3> svd(qpT.template topLeftCorner<3, 3>()-qpT.template topRightCorner<3, 1>()*qpT.template bottomLeftCorner<1, 3>(), Eigen::ComputeFullU|Eigen::ComputeFullV);
				Matrix3 d=Matrix3::Identity();
				d(2, 2)=(svd.matrixU()*svd.matrixV().transpose()).determinant();
				cluster_transform.rotMat(k, 0)=svd.matrixU()*d*svd.matrixV().transpose();
				cluster_transform.transVec(k, 0)=qpT.template topRightCorner<3, 1>()-cluster_transform.rotMat(k, 0)*qpT.template bottomLeftCorner<1, 3>().transpose();
			}
			
			// std::cout << "Cluster Transform:" << qpT << std::endl;
			// std::cout << "Cluster Translation:" << cluster_transform.transVec(k,0) << std::endl;
			// std::cout << "Cluster Rotation:" << cluster_transform.rotMat(k,0) << std::endl;

			for (int ind=0; ind<vert_inds.size(); ind++) {
				int i = vert_inds[ind];
				// if(par==false)
				// 	std::cout << "V':" << cluster_transform.rotMat(k, 0)*u.vec3(subjectID(k), i) << cluster_transform.transVec(k, 0) << v.vec3(k, i).template cast<_Scalar>() << std::endl;
				#pragma omp atomic
				cluster_error+=(cluster_transform.rotMat(k, 0)*u.vec3(subjectID(k), i)+cluster_transform.transVec(k, 0)-v.vec3(k, i).template cast<_Scalar>()).squaredNorm();
			}
		}

		cluster_error /= nF;

		return cluster_error;
	}





	/** Update bone transformation from label
	*/
	void computeTransFromLabel() {
		// std::cout << "Computing Trans for labels. nB:" << nB  << std::endl;
		m=Matrix4::Identity().replicate(nF, nB);
		#pragma omp parallel for
		for (int k=0; k<nF; k++) {
			MatrixX qpT=MatrixX::Zero(4, 4*nB);

			for (int i=0; i<nV; i++) 
				if (label(i)!=-1) qpT.blk4(0, label(i))+=Vector4(v.vec3(k, i).template cast<_Scalar>().homogeneous())*u.vec3(subjectID(k), i).homogeneous().transpose();
			for (int j=0; j<nB; j++) qpT2m(qpT.blk4(0, j), k, j);
		}
	}

	/** Set matrix w from label
	*/
	void labelToWeights() {
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> trip(nV);
		for (int i=0; i<nV; i++) trip[i]=Triplet(label(i), i, _Scalar(1));
		w.resize(nB, nV);
		w.setFromTriplets(trip.begin(), trip.end());
		lockW=VectorX::Zero(nV);
	}

	/** Split bone clusters
		@param maxB is the maximum number of bones
		@param threshold*2 is the minimum size of the bone cluster to be splited 
	*/
	void split(int maxB, int threshold) {
		//Centroids
		MatrixX cu=MatrixX::Zero(3*nS, nB);
		Eigen::VectorXi s=Eigen::VectorXi::Zero(nB);
		for (int i=0; i<nV; i++) {
			cu.col(label(i))+=u.col(i);
			s(label(i))++;
		}
		for (int j=0; j<nB; j++) if (s(j)!=0) cu.col(j)/=_Scalar(s(j));

		//Distance to centroid & error
		VectorX d(nV), e(nV);
		VectorX minD=VectorX::Constant(nB, std::numeric_limits<_Scalar>::max());
		VectorX minE=VectorX::Constant(nB, std::numeric_limits<_Scalar>::max());
		VectorX ce=VectorX::Zero(nB);

		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			int j=label(i);
			d(i)=(u.col(i)-cu.col(j)).norm();
			e(i)=sqrt(errorVtxBone(i, j, false));
			if (d(i)<minD(j)) {
				#pragma omp critical
				minD(j)=std::min(minD(j), d(i));
			}
			if (e(i)<minE(j)) {
				#pragma omp critical
				minE(j)=std::min(minE(j), e(i));
			}
			#pragma omp atomic
			ce(j)+=e(i);
		}

		//Seed
		Eigen::VectorXi seed=Eigen::VectorXi::Constant(nB, -1);
		VectorX gMax(nB);

		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			int j=label(i);
			double tmp=abs((e(i)-minE(j))*(d(i)-minD(j)));

			if ((seed(j)==-1)||(tmp>gMax(j))) {
				#pragma omp critical
				if ((seed(j)==-1)||(tmp>gMax(j))) {
					gMax(j)=tmp;
					seed(j)=i;
				}
			}
		}

		int countID=nB;
		std::cout << "Spliting Clusters: Orignal:" << nB << std::endl; 
		_Scalar avgErr=ce.sum()/nB;
		for (int j=0; j<nB; j++){
			//std::cout << "Can split cluster: Number of vertices constrint:" << (s(j)>threshold*2) << " Avg Error constraint:" <<  (ce(j)>avgErr/100) << std::endl;
			if ((countID<maxB)&&(s(j)>threshold*2)&&(ce(j)>avgErr/100)) {
				int newLabel=countID++;
				int i=seed(j);
				label(i) = newLabel;
				if (splitSeedNeighbors>0) {
					int assigned=0;
					if (fv.empty() || fv.size()==0) {
						computeKNNGraph(std::max(8, splitSeedNeighbors));
						for (const auto& neighbor: cached_knn_graph[i]) {
							if (assigned>=splitSeedNeighbors) break;
							if (label(neighbor.second)==j) {
								label(neighbor.second)=newLabel;
								assigned++;
							}
						}
					} else {
						if (laplacian.cols()!=nV) computeSmoothSolver();
						for (typename SparseMatrix::InnerIterator it(laplacian, i); it; ++it) {
							if (assigned>=splitSeedNeighbors) break;
							if (label(it.row())==j) {
								label(it.row())=newLabel;
								assigned++;
							}
						}
					}
				}
			}
		}
		nB=countID;
		std::cout << "Spliting Clusters: New:" << nB << std::endl; 

	}

	/** Remove bones with small number of associated vertices
		@param threshold is the minimum number of vertices assigned to a bone
	*/
	void pruneBones(int threshold) {
		Eigen::VectorXi s=Eigen::VectorXi::Zero(nB);
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			#pragma omp atomic
			s(label(i))++;
		}

		Eigen::VectorXi newID(nB);
		int countID=0;
		for (int j=0; j<nB; j++)
			if (s(j)<threshold) newID(j)=-1; else newID(j)=countID++;

		if (countID==nB) return;

		for (int j=0; j<nB; j++)
			if (newID(j)!=-1) m.template middleCols<4>(newID(j)*4)=m.template middleCols<4>(j*4);

		#pragma omp parallel for
		for (int i=0; i<nV; i++) label(i)=newID(label(i));

		nB=countID;
		m.conservativeResize(nF*4, nB*4);
		reassignLabels();
	}

	/** Initialize skinning weights with rigid bind to the best bone
	*/
	void initWeights() {
		label=Eigen::VectorXi::Constant(nV, -1);
		#pragma omp parallel for
		for (int i=0; i<nV; i++) {
			_Scalar gMin;
			for (int j=0; j<nB; j++) {
				_Scalar ej=errorVtxBone(i, j, false);
				if ((label(i)==-1)||(gMin>ej)) {
					gMin=ej;
					label(i)=j;
				}
			}
		}
		reassignLabels();
		labelToWeights();
	}

	//! vuT.blk4(k, j) = \sum_{i=0}^{nV-1}  w(j, i)*v.vec3(k, i).homogeneous()*u.vec3(subjectID(k), i).homogeneous()^T
	MatrixX vuT;

	/** Pre-compute vuT with bone translations affinity soft constraint
	*/
	void compute_vuT() {
		vuT=MatrixX::Zero(nF*4, nB*4);
		#pragma omp parallel for
		for (int k=0; k<nF; k++) {
			MatrixX vuTp=MatrixX::Zero(4, nB*4);
			for (int i=0; i<nV; i++)
				for (typename SparseMatrix::InnerIterator it(w, i); it; ++it) {
					Matrix4 tmp=Vector4(v.vec3(k, i).template cast<_Scalar>().homogeneous())*u.vec3(subjectID(k), i).homogeneous().transpose();
					vuT.blk4(k, it.row())+=it.value()*tmp;
					vuTp.blk4(0, it.row())+=pow(it.value(), transAffineNorm)*tmp;
				}
			for (int j=0; j<nB; j++)
				if (vuTp(3, j*4+3)!=0)
					vuT.blk4(k, j)+=(transAffine*vuT(k*4+3, j*4+3)/vuTp(3, j*4+3))*vuTp.blk4(0, j);
		}
	}
	
	//! uuT is a sparse block matrix, uuT(j, k).block<4, 4>(s*4, 0) = \sum{i=0}{nV-1} w(j, i)*w(k, i)*u.col(i).segment<3>(s*3).homogeneous().transpose()*u.col(i).segment<3>(s*3).homogeneous()
	struct SparseMatrixBlock {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		MatrixX val;
		Eigen::VectorXi innerIdx, outerIdx;
	} uuT;

	/** Pre-compute uuT for bone transformations update
	*/
	void compute_uuT() {
		Eigen::MatrixXi pos=Eigen::MatrixXi::Constant(nB, nB, -1);
		#pragma omp parallel for
		for (int i=0; i<nV; i++)
			for (typename SparseMatrix::InnerIterator it(w, i); it; ++it)
				for (typename SparseMatrix::InnerIterator jt(w, i); jt; ++jt)
					pos(it.row(), jt.row())=1;

		uuT.outerIdx.resize(nB+1);
		uuT.innerIdx.resize(nB*nB);
		int nnz=0;
		for (int j=0; j<nB; j++) {
			uuT.outerIdx(j)=nnz;
			for (int i=0; i<nB; i++)
				if (pos(i, j)!=-1) {
					uuT.innerIdx(nnz)=i;
					pos(i, j)=nnz++;
				}
		}
		uuT.outerIdx(nB)=nnz;
		uuT.innerIdx.conservativeResize(nnz);
		uuT.val=MatrixX::Zero(nS*4, nnz*4);

		#pragma omp parallel for
		for (int i=0; i<nV; i++)
			for (typename SparseMatrix::InnerIterator it(w, i); it; ++it)
				for (typename SparseMatrix::InnerIterator jt(w, i); jt; ++jt)
					if (it.row()>=jt.row()) {
						double _w=it.value()*jt.value();
						MatrixX _uuT(4*nS, 4);
						Vector4 _u;
						for (int s=0; s<nS; s++) {
							_u=u.vec3(s, i).homogeneous();
							_uuT.blk4(s, 0)=_w*_u*_u.transpose();
						}
						int p=pos(it.row(), jt.row())*4;
						for (int c=0; c<4; c++)
							for (int r=0; r<4*nS; r++)
								#pragma omp atomic
								uuT.val(r, p+c)+=_uuT(r, c);
					}

		for (int i=0; i<nB; i++)
			for (int j=i+1; j<nB; j++)
				if (pos(i, j)!=-1)
					uuT.val.middleCols(pos(i, j)*4, 4)=uuT.val.middleCols(pos(j, i)*4, 4);
	}



	/** Pre-compute mTm for weights update
	*/
	void compute_mTm() {
		Eigen::MatrixXi idx(2, nB*(nB+1)/2);
		int nPairs=0;
		for (int i=0; i<nB; i++)
			for (int j=i; j<nB; j++) {
				idx(0, nPairs)=i;
				idx(1, nPairs)=j;
				nPairs++;
			}

		mTm=MatrixX::Zero(nS*nB*4, nB*4);
		#pragma omp parallel for
		for (int p=0; p<nPairs; p++) {
			int i=idx(0, p);
			int j=idx(1, p);
			for (int k=0; k<nF; k++)
				mTm.blk4(subjectID(k)*nB+i, j)+=m.blk4(k, i).template topRows<3>().transpose()*m.blk4(k, j).template topRows<3>();
			if (i!=j) for (int s=0; s<nS; s++) mTm.blk4(s*nB+j, i)=mTm.blk4(s*nB+i, j);
		}
	}

	//! aTb.col(i) is the A^Tb for vertex i, where A.size = (3*nF, nB), A.col(j).segment<3>(f*3) is the transformed position of vertex i by bone j at frame f, b = v.col(i).
	MatrixX aTb;

	/** Pre-compute aTb for weights update
	*/
	void compute_aTb() {
		// int compute_count=0;
		#pragma omp parallel for
		for (int i=0; i<nV; i++)
			for (int j=0; j<nB; j++)
				if ((aTb(j, i)==0)&&(ws(j, i)>weightEps)){
					// compute_count++;
					for (int k=0; k<nF; k++)
						aTb(j, i)+=v.vec3(k, i).template cast<_Scalar>().dot(m.blk4(k, j).template topRows<3>()*u.vec3(subjectID(k), i).homogeneous());
				}
		// std::cout << "compute_aTb: " << compute_count << "should_compute: " << nV*nB << std::endl;
	}

	//! Size of the model=RMS distance to centroid
	_Scalar modelSize;
	
	//! Laplacian matrix
	SparseMatrix laplacian;

	//! LU factorization of Laplacian
	Eigen::SparseLU<SparseMatrix> smoothSolver;

	/** Pre-compute Laplacian and LU factorization
	*/
	void computeSmoothSolver() {
		int nFV=(int)fv.size();

		_Scalar epsDis=0;
		for (int f=0; f<nFV; f++) {
			int nf=(int)fv[f].size();
			for (int g=0; g<nf; g++) {
				int i=fv[f][g];
				int j=fv[f][(g+1)%nf];
				epsDis+=(u.col(i)-u.col(j)).norm();
			}
		}
		epsDis=epsDis*weightEps/(_Scalar)nS;

		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> triplet;
		VectorX d=VectorX::Zero(nV);
		std::vector<std::set<int>> isComputed(nV);

		#pragma omp parallel for
		for (int f=0; f<nFV; f++) {
			int nf=(int)fv[f].size();
			for (int g=0; g<nf; g++) {
				int i=fv[f][g];
				int j=fv[f][(g+1)%nf];

				bool needCompute=false;
				#pragma omp critical 
				if (isComputed[i].find(j)==isComputed[i].end()) {
					needCompute=true;
					isComputed[i].insert(j);
					isComputed[j].insert(i);
				}

				if (needCompute) {
					double val=0;
					for (int s=0; s<nS; s++) {
						double du=(u.vec3(s, i)-u.vec3(s, j)).norm();
						for (int k=fStart(s); k<fStart(s+1); k++)
							val+=pow((v.vec3(k, i).template cast<_Scalar>()-v.vec3(k, j).template cast<_Scalar>()).norm()-du, 2);
					}
					val=1/(sqrt(val/nF)+epsDis);

					#pragma omp critical
					triplet.push_back(Triplet(i, j, -val));
					#pragma omp atomic
					d(i)+=val;

					#pragma omp critical
					triplet.push_back(Triplet(j, i, -val));
					#pragma omp atomic
					d(j)+=val;
				}
			}
		}

		for (int i=0; i<nV; i++)
			triplet.push_back(Triplet(i, i, d(i)));

		laplacian.resize(nV, nV);
		laplacian.setFromTriplets(triplet.begin(), triplet.end());

		for (int i=0; i<nV; i++)
			if (d(i)!=0) laplacian.row(i)/=d(i);

		laplacian=weightsSmoothStep*laplacian+SparseMatrix((VectorX::Ones(nV)).asDiagonal());
		smoothSolver.compute(laplacian);
	}

	//! K-nearest neighbors for KNN-based smoothing
	int knn_neighbors;

	//! 缓存的KNN邻接关系
	std::vector<std::vector<std::pair<_Scalar, int>>> cached_knn_graph;
	bool knn_graph_computed;

	//! KNN adjacency matrix
	SparseMatrix knn_laplacian;

	//! LU factorization of KNN Laplacian  
	Eigen::SparseLU<SparseMatrix> knn_smoothSolver;

	/** @brief 计算并缓存KNN邻接关系
		@param k 每个顶点的最近邻数量 (默认: 8)
	*/
	void computeKNNGraph(int k = 8) {
		if (knn_graph_computed && knn_neighbors == k) {
			// 如果已经计算过相同参数的KNN图，直接返回
			return;
		}
		
		knn_neighbors = k;
		cached_knn_graph.resize(nV);
		
		std::cout << "init KNN, k=" << k << std::endl;
		
		#pragma omp parallel for
		for (int i = 0; i < nV; i++) {
			std::priority_queue<std::pair<_Scalar, int>> nearest;
			
			// 计算到所有其他顶点的距离
			for (int j = 0; j < nV; j++) {
				if (i != j) {
					_Scalar dist = (u.vec3(0, i) - u.vec3(0, j)).norm();
					if ((int)nearest.size() < k) {
						nearest.push(std::make_pair(dist, j));
					} else if (dist < nearest.top().first) {
						nearest.pop();
						nearest.push(std::make_pair(dist, j));
					}
				}
			}
			
			std::vector<std::pair<_Scalar, int>> neighbors;
			neighbors.reserve(k);
			while (!nearest.empty()) {
				neighbors.push_back(nearest.top());
				nearest.pop();
			}
			cached_knn_graph[i] = std::move(neighbors);
		}
		
		knn_graph_computed = true;
		std::cout << "init KNN, k=" << k << " done" << std::endl;
	}

	/** @brief 基于缓存的KNN图构建拉普拉斯矩阵和LU分解
		@param k 每个顶点的最近邻数量 (默认: 8)
		@param sigma 高斯核宽度参数 (默认: 自动计算)
	*/
	void computeKNNSmoothSolver(int k = 8, _Scalar sigma = -1) {
		// 1. 确保KNN图已计算并缓存
		computeKNNGraph(k);
		
		// 2. 自动计算sigma (如果需要)
		if (sigma < 0) {
			_Scalar total_distance = 0;
			int edge_count = 0;
			
			for (int i = 0; i < nV; i++) {
				for (const auto& pair : cached_knn_graph[i]) {
					total_distance += pair.first;
					edge_count++;
				}
			}
			
			sigma = total_distance / edge_count;
			// std::cout << "自动计算的sigma: " << sigma << std::endl;
		}
		
		// 3. 基于缓存的KNN图构建拉普拉斯矩阵
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> triplet;
		triplet.reserve(nV * k * 2 + nV);  // 预分配空间
		
		VectorX row_sum = VectorX::Zero(nV);
		
		
		for (int i = 0; i < nV; i++) {
			for (const auto& pair : cached_knn_graph[i]) {
				_Scalar dist = pair.first;
				int j = pair.second;
				
				// 高斯核权重
				_Scalar weight = std::exp(-dist * dist / (2 * sigma * sigma));
				
				// 添加边 (i, j)，确保对称性
				triplet.push_back(Triplet(i, j, -weight));
				triplet.push_back(Triplet(j, i, -weight));
				
				row_sum(i) += weight;
				row_sum(j) += weight;
			}
		}
		
		// 4. 添加对角元素
		for (int i = 0; i < nV; i++) {
			triplet.push_back(Triplet(i, i, row_sum(i)));
		}
		
		// 5. 构建稀疏矩阵
		knn_laplacian.resize(nV, nV);
		knn_laplacian.setFromTriplets(triplet.begin(), triplet.end());
		
		// 6. 行归一化
		for (int i = 0; i < nV; i++) {
			if (row_sum(i) > 1e-10) {
				knn_laplacian.row(i) /= row_sum(i);
			}
		}
		
		// 7. 添加身份矩阵并计算LU分解
		knn_laplacian = weightsSmoothStep * knn_laplacian + 
						SparseMatrix((VectorX::Ones(nV)).asDiagonal());
		
		knn_smoothSolver.compute(knn_laplacian);
		
		if (knn_smoothSolver.info() != Eigen::Success) {
			std::cout << "警告: KNN拉普拉斯矩阵LU分解失败!" << std::endl;
		} else {
			std::cout << "KNN solved, k=" << k << ", sigma=" << sigma;
		}
	}

	/** @brief KNN-based weight smoothing
	*/
	void compute_ws_knn() {
		ws = w.transpose();
		
		// 使用KNN平滑求解器
		#pragma omp parallel for
		for (int j = 0; j < nB; j++) {
			if (knn_smoothSolver.info() == Eigen::Success) {
				ws.col(j) = knn_smoothSolver.solve(ws.col(j));
			}
		}
		
		ws.transposeInPlace();

		// 后处理：确保非负性和归一化
		#pragma omp parallel for
		for (int i = 0; i < nV; i++) {
			ws.col(i) = ws.col(i).cwiseMax(0.0);
			_Scalar si = ws.col(i).sum();
			if (si < _Scalar(0.1)) {
				ws.col(i) = VectorX::Constant(nB, _Scalar(1) / nB);
			} else {
				ws.col(i) /= si;
			}
		}
	}


	//! Smoothed skinning weights
	MatrixX ws;

	/** Implicit skinning weights Laplacian smoothing
	*/
	void compute_ws() {
		if (fv.empty() || fv.size() == 0) {
			if (pointCloudSmoothMode==0) {
				ws=MatrixX(w);
			} else {
				compute_ws_knn();
			}
    } else {
        // 网格数据：使用原来的平滑方法
        ws = w.transpose();
        #pragma omp parallel for
        for (int j = 0; j < nB; j++) {
            ws.col(j) = smoothSolver.solve(ws.col(j));
        }
        ws.transposeInPlace();

        #pragma omp parallel for
        for (int i = 0; i < nV; i++) {
            ws.col(i) = ws.col(i).cwiseMax(0.0);
            _Scalar si = ws.col(i).sum();
            if (si < _Scalar(0.1)) ws.col(i) = VectorX::Constant(nB, _Scalar(1)/nB); 
            else ws.col(i) /= si;
        }
    }
}

	//! Per-vertex weights solver
	ConvexLS<_Scalar> wSolver;

	/** Pre-compute aTa for weights update on one vertex
		@param i is the vertex index.
		@param aTa is the by-reference output of A^TA for vertex i, where A.size = (3*nF, nB), A.col(j).segment<3>(f*3) is the transformed position of vertex i by bone j at frame f.
	*/
	void compute_aTa(int i, MatrixX& aTa) {
		aTa=MatrixX::Zero(nB, nB);
		for (int j1=0; j1<nB; j1++)
			for (int j2=j1; j2<nB; j2++) {
				for (int s=0; s<nS; s++) aTa(j1, j2)+=u.vec3(s, i).homogeneous().dot(mTm.blk4(s*nB+j1, j2)*u.vec3(s, i).homogeneous());
				if (j1!=j2) aTa(j2, j1)=aTa(j1, j2);
			}
	}
};
	
}

#ifdef DEM_BONES_DEM_BONES_MAT_BLOCKS_UNDEFINED
#undef blk4
#undef rotMat
#undef transVec
#undef vec3
#undef DEM_BONES_MAT_BLOCKS
#endif

#endif
