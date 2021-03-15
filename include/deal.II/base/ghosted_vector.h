// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_ghosted_vector_h
#define dealii_ghosted_vector_h


#include <deal.II/base/config.h>

#include <deal.II/base/communication_pattern_base.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/memory_space_data.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/thread_management.h>

#include <iomanip>
#include <memory>


DEAL_II_NAMESPACE_OPEN


/*! @addtogroup Vectors
 *@{
 */

/**
 * Implementation of a parallel vector class. The design of this class is
 * similar to the standard Vector class in deal.II, with the
 * exception that storage is distributed with MPI.
 *
 * The vector is designed for the following scheme of parallel
 * partitioning:
 * <ul>
 * <li> The indices held by individual processes (locally owned part) in
 * the MPI parallelization form a contiguous range
 * <code>[my_first_index,my_last_index)</code>.
 * <li> Ghost indices residing on arbitrary positions of other processors
 * are allowed. It is in general more efficient if ghost indices are
 * clustered, since they are stored as a set of intervals. The
 * communication pattern of the ghost indices is determined when calling
 * the function <code>reinit (locally_owned, ghost_indices,
 * communicator)</code>, and retained until the partitioning is changed.
 * This allows for efficient parallel communication of indices. In
 * particular, it stores the communication pattern, rather than having to
 * compute it again for every communication. For more information on ghost
 * vectors, see also the
 * @ref GlossGhostedVector "glossary entry on vectors with ghost elements".
 * <li> Besides the usual global access operator() it is also possible to
 * access vector entries in the local index space with the function @p
 * local_element(). Locally owned indices are placed first, [0,
 * locally_owned_size()), and then all ghost indices follow after them
 * contiguously, [locally_owned_size(),
 * locally_owned_size()+n_ghost_entries()).
 * </ul>
 *
 * Functions related to parallel functionality:
 * <ul>
 * <li> The function <code>compress()</code> goes through the data
 * associated with ghost indices and communicates it to the owner process,
 * which can then add it to the correct position. This can be used e.g.
 * after having run an assembly routine involving ghosts that fill this
 * vector. Note that the @p insert mode of @p compress() does not set the
 * elements included in ghost entries but simply discards them, assuming
 * that the owning processor has set them to the desired value already
 * (See also the
 * @ref GlossCompress "glossary entry on compress").
 * <li> The <code>update_ghost_values()</code> function imports the data
 * from the owning processor to the ghost indices in order to provide read
 * access to the data associated with ghosts.
 * <li> It is possible to split the above functions into two phases, where
 * the first initiates the communication and the second one finishes it.
 * These functions can be used to overlap communication with computations
 * in other parts of the code.
 * <li> Of course, reduction operations (like norms) make use of
 * collective all-to-all MPI communications.
 * </ul>
 *
 * This vector can take two different states with respect to ghost
 * elements:
 * <ul>
 * <li> After creation and whenever zero_out_ghost_values() is called (or
 * <code>operator= (0.)</code>), the vector does only allow writing into
 * ghost elements but not reading from ghost elements.
 * <li> After a call to update_ghost_values(), the vector does not allow
 * writing into ghost elements but only reading from them. This is to
 * avoid undesired ghost data artifacts when calling compress() after
 * modifying some vector entries. The current status of the ghost entries
 * (read mode or write mode) can be queried by the method
 * has_ghost_elements(), which returns <code>true</code> exactly when
 * ghost elements have been updated and <code>false</code> otherwise,
 * irrespective of the actual number of ghost entries in the vector layout
 * (for that information, use n_ghost_entries() instead).
 * </ul>
 *
 * This vector uses the facilities of the class dealii::Vector<Number> for
 * implementing the operations on the local range of the vector. In
 * particular, it also inherits thread parallelism that splits most
 * vector-vector operations into smaller chunks if the program uses
 * multiple threads. This may or may not be desired when working also with
 * MPI.
 *
 * <h4>Limitations regarding the vector size</h4>
 *
 * This vector class is based on two different number types for indexing.
 * The so-called global index type encodes the overall size of the vector.
 * Its type is types::global_dof_index. The largest possible value is
 * <code>2^32-1</code> or approximately 4 billion in case 64 bit integers
 * are disabled at configuration of deal.II (default case) or
 * <code>2^64-1</code> or approximately <code>10^19</code> if 64 bit
 * integers are enabled (see the glossary entry on
 * @ref GlobalDoFIndex
 * for further information).
 *
 * The second relevant index type is the local index used within one MPI
 * rank. As opposed to the global index, the implementation assumes 32-bit
 * unsigned integers unconditionally. In other words, to actually use a
 * vector with more than four billion entries, you need to use MPI with
 * more than one rank (which in general is a safe assumption since four
 * billion entries consume at least 16 GB of memory for floats or 32 GB of
 * memory for doubles) and enable 64-bit indices. If more than 4 billion
 * local elements are present, the implementation tries to detect that,
 * which triggers an exception and aborts the code. Note, however, that
 * the detection of overflow is tricky and the detection mechanism might
 * fail in some circumstances. Therefore, it is strongly recommended to
 * not rely on this class to automatically detect the unsupported case.
 *
 * <h4>CUDA support</h4>
 *
 * This vector class supports two different memory spaces: Host and CUDA. By
 * default, the memory space is Host and all the data are allocated on the
 * CPU. When the memory space is CUDA, all the data is allocated on the GPU.
 * The operations on the vector are performed on the chosen memory space. *
 * From the host, there are two methods to access the elements of the Vector
 * when using the CUDA memory space:
 * <ul>
 * <li> use get_values():
 * @code
 * Vector<double, MemorySpace::CUDA> vector(local_range, comm);
 * double* vector_dev = vector.get_values();
 * std::vector<double> vector_host(local_range.n_elements(), 1.);
 * Utilities::CUDA::copy_to_dev(vector_host, vector_dev);
 * @endcode
 * <li> use import():
 * @code
 * Vector<double, MemorySpace::CUDA> vector(local_range, comm);
 * ReadWriteVector<double> rw_vector(local_range);
 * for (auto & val : rw_vector)
 *   val = 1.;
 * vector.import(rw_vector, VectorOperations::insert);
 * @endcode
 * </ul>
 * The import method is a lot safer and will perform an MPI communication if
 * necessary. Since an MPI communication may be performed, import needs to
 * be called on all the processors.
 *
 * @note By default, all the ranks will try to access the device 0. This is
 * fine is if you have one rank per node and one gpu per node. If you
 * have multiple GPUs on one node, we need each process to access a
 * different GPU. If each node has the same number of GPUs, this can be done
 * as follows:
 * <code> int n_devices = 0; cudaGetDeviceCount(&n_devices); int
 * device_id = my_rank % n_devices;
 * cudaSetDevice(device_id);
 * </code>
 *
 * <h4>MPI-3 shared-memory support</h4>
 *
 * In Host mode, this class allows to use MPI-3 shared-memory features
 * by providing a separate MPI communicator that consists of processes on
 * the same shared-memory domain. By calling
 * <code> vector.shared_vector_data();
 * </code>
 * users have read-only access to both locally-owned and ghost values of
 * processes combined in the shared-memory communicator (@p comm_sm in
 * reinit()).
 *
 * You can create a communicator consisting of all processes on
 * the same shared-memory domain with:
 * <code> MPI_Comm comm_sm;
 * MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
 * &comm_sm);
 * </code>
 *
 * @see CUDAWrappers
 */
template <typename Number, typename MemorySpaceType = MemorySpace::Host>
class GhostedVector : public Subscriptor
{
public:
  using memory_space    = MemorySpaceType;
  using value_type      = Number;
  using pointer         = value_type *;
  using const_pointer   = const value_type *;
  using iterator        = value_type *;
  using const_iterator  = const value_type *;
  using reference       = value_type &;
  using const_reference = const value_type &;
  using size_type       = types::global_dof_index;
  using real_type       = typename numbers::NumberTraits<Number>::real_type;

  static_assert(std::is_same<MemorySpaceType, MemorySpace::Host>::value ||
                  std::is_same<MemorySpaceType, MemorySpace::CUDA>::value,
                "MemorySpace should be Host or CUDA");

  /**
   * @name 1: Basic Object-handling
   */
  //@{
  /**
   * Empty constructor.
   */
  GhostedVector();

  /**
   * Copy constructor. Uses the parallel partitioning of @p in_vector.
   * It should be noted that this constructor automatically sets ghost
   * values to zero. Call @p update_ghost_values() directly following
   * construction if a ghosted vector is required.
   */
  GhostedVector(const GhostedVector<Number, MemorySpaceType> &in_vector);

  /**
   * Construct a parallel vector of the given global size without any
   * actual parallel distribution.
   */
  GhostedVector(const size_type size);

  /**
   * Construct a parallel vector. The local range is specified by @p
   * locally_owned_set (note that this must be a contiguous interval,
   * multiple intervals are not possible). The IndexSet @p ghost_indices
   * specifies ghost indices, i.e., indices which one might need to read
   * data from or accumulate data from. It is allowed that the set of
   * ghost indices also contains the local range, but it does not need to.
   *
   * This function involves global communication, so it should only be
   * called once for a given layout. Use the constructor with
   * Vector<Number> argument to create additional vectors with the same
   * parallel layout.
   *
   * @see
   * @ref GlossGhostedVector "vectors with ghost elements"
   */
  GhostedVector(const IndexSet &local_range,
                const IndexSet &ghost_indices,
                const MPI_Comm &communicator);

  /**
   * Same constructor as above but without any ghost indices.
   */
  GhostedVector(const IndexSet &local_range, const MPI_Comm &communicator);

  /**
   * Create the vector based on the parallel partitioning described in @p
   * partitioner. The input argument is a shared pointer, which stores the
   * partitioner data only once and share it between several vectors with
   * the same layout.
   */
  GhostedVector(
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner);

  /**
   * Destructor.
   */
  virtual ~GhostedVector() override;

  /**
   * Set the global size of the vector to @p size without any actual
   * parallel distribution.
   */
  void
  reinit(const size_type size, const bool omit_zeroing_entries = false);

  /**
   * Uses the parallel layout of the input vector @p in_vector and
   * allocates memory for this vector. Recommended initialization function
   * when several vectors with the same layout should be created.
   *
   * If the flag @p omit_zeroing_entries is set to false, the memory will
   * be initialized with zero, otherwise the memory will be untouched (and
   * the user must make sure to fill it with reasonable data before using
   * it).
   */
  template <typename Number2>
  void
  reinit(const GhostedVector<Number2, MemorySpaceType> &in_vector,
         const bool omit_zeroing_entries = false);

  /**
   * Initialize the vector. The local range is specified by @p
   * locally_owned_set (note that this must be a contiguous interval,
   * multiple intervals are not possible). The IndexSet @p ghost_indices
   * specifies ghost indices, i.e., indices which one might need to read
   * data from or accumulate data from. It is allowed that the set of
   * ghost indices also contains the local range, but it does not need to.
   *
   * This function involves global communication, so it should only be
   * called once for a given layout. Use the @p reinit function with
   * Vector<Number> argument to create additional vectors with the same
   * parallel layout.
   *
   * @see
   * @ref GlossGhostedVector "vectors with ghost elements"
   */
  void
  reinit(const IndexSet &local_range,
         const IndexSet &ghost_indices,
         const MPI_Comm &communicator);

  /**
   * Same as above, but without ghost entries.
   */
  void
  reinit(const IndexSet &local_range, const MPI_Comm &communicator);

  /**
   * Initialize the vector given to the parallel partitioning described in
   * @p partitioner. The input argument is a shared pointer, which stores
   * the partitioner data only once and share it between several vectors
   * with the same layout.
   *
   * The optional argument @p comm_sm, which consists of processes on
   * the same shared-memory domain, allows users have read-only access to
   * both locally-owned and ghost values of processes combined in the
   * shared-memory communicator.
   */
  void
  reinit(const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
         const MPI_Comm &comm_sm = MPI_COMM_SELF);

  /**
   * Initialize vector with @p local_size locally-owned and @p ghost_size
   * ghost degrees of freedoms.
   *
   * The optional argument @p comm_sm, which consists of processes on
   * the same shared-memory domain, allows users have read-only access to
   * both locally-owned and ghost values of processes combined in the
   * shared-memory communicator.
   *
   * @note In the created underlying partitioner, the local index range is
   *   translated to global indices in an ascending and one-to-one fashion,
   *   i.e., the indices of process $p$ sit exactly between the indices of
   *   the processes $p-1$ and $p+1$, respectively. Setting the
   *   @p ghost_size variable to an appropriate value provides memory space
   *   for the ghost data in a vector's memory allocation as and allows
   *   access to it via local_element(). However, the associated global
   *   indices must be handled externally in this case.
   */
  void
  reinit(const types::global_dof_index local_size,
         const types::global_dof_index ghost_size,
         const MPI_Comm &              comm,
         const MPI_Comm &              comm_sm = MPI_COMM_SELF);

  /**
   * Swap the contents of this vector and the other vector @p v. One could
   * do this operation with a temporary variable and copying over the data
   * elements, but this function is significantly more efficient since it
   * only swaps the pointers to the data of the two vectors and therefore
   * does not need to allocate temporary storage and move data around.
   *
   * This function is analogous to the @p swap function of all C++
   * standard containers. Also, there is a global function
   * <tt>swap(u,v)</tt> that simply calls <tt>u.swap(v)</tt>, again in
   * analogy to standard functions.
   */
  void
  swap(GhostedVector<Number, MemorySpaceType> &v);

  /**
   * Assigns the vector to the parallel partitioning of the input vector
   * @p in_vector, and copies all the data.
   *
   * If one of the input vector or the calling vector (to the left of the
   * assignment operator) had ghost elements set before this operation,
   * the calling vector will have ghost values set. Otherwise, it will be
   * in write mode. If the input vector does not have any ghost elements
   * at all, the vector will also update its ghost values in analogy to
   * the respective setting the Trilinos and PETSc vectors.
   */
  GhostedVector<Number, MemorySpaceType> &
  operator=(const GhostedVector<Number, MemorySpaceType> &in_vector);

  /**
   * Assigns the vector to the parallel partitioning of the input vector
   * @p in_vector, and copies all the data.
   *
   * If one of the input vector or the calling vector (to the left of the
   * assignment operator) had ghost elements set before this operation,
   * the calling vector will have ghost values set. Otherwise, it will be
   * in write mode. If the input vector does not have any ghost elements
   * at all, the vector will also update its ghost values in analogy to
   * the respective setting the Trilinos and PETSc vectors.
   */
  template <typename Number2>
  GhostedVector<Number, MemorySpaceType> &
  operator=(const GhostedVector<Number2, MemorySpaceType> &in_vector);

  //@}

  /**
   * @name 2: Parallel data exchange
   */
  //@{
  /**
   * This function copies the data that has accumulated in the data buffer
   * for ghost indices to the owning processor. For the meaning of the
   * argument @p operation, see the entry on
   * @ref GlossCompress "Compressing distributed vectors and matrices"
   * in the glossary.
   *
   * There are four variants for this function. If called with argument @p
   * VectorOperation::add adds all the data accumulated in ghost elements
   * to the respective elements on the owning processor and clears the
   * ghost array afterwards. If called with argument @p
   * VectorOperation::insert, a set operation is performed. Since setting
   * elements in a vector with ghost elements is ambiguous (as one can set
   * both the element on the ghost site as well as the owning site), this
   * operation makes the assumption that all data is set correctly on the
   * owning processor. Upon call of compress(VectorOperation::insert), all
   * ghost entries are thus simply zeroed out (using zero_ghost_values()).
   * In debug mode, a check is performed for whether the data set is
   * actually consistent between processors, i.e., whenever a non-zero
   * ghost element is found, it is compared to the value on the owning
   * processor and an exception is thrown if these elements do not agree.
   * If called with VectorOperation::min or VectorOperation::max, the
   * minimum or maximum on all elements across the processors is set.
   * @note This vector class has a fixed set of ghost entries attached to
   * the local representation. As a consequence, all ghost entries are
   * assumed to be valid and will be exchanged unconditionally according
   * to the given VectorOperation. Make sure to initialize all ghost
   * entries with the neutral element of the given VectorOperation or
   * touch all ghost entries. The neutral element is zero for
   * VectorOperation::add and VectorOperation::insert, `+inf` for
   * VectorOperation::min, and `-inf` for VectorOperation::max. If all
   * values are initialized with values below zero and compress is called
   * with VectorOperation::max two times subsequently, the maximal value
   * after the second calculation will be zero.
   */
  virtual void
  compress(VectorOperation::values operation);

  /**
   * Fills the data field for ghost indices with the values stored in the
   * respective positions of the owning processor. This function is needed
   * before reading from ghosts. The function is @p const even though
   * ghost data is changed. This is needed to allow functions with a @p
   * const vector to perform the data exchange without creating
   * temporaries.
   *
   * After calling this method, write access to ghost elements of the
   * vector is forbidden and an exception is thrown. Only read access to
   * ghost elements is allowed in this state. Note that all subsequent
   * operations on this vector, like global vector addition, etc., will
   * also update the ghost values by a call to this method after the
   * operation. However, global reduction operations like norms or the
   * inner product will always ignore ghost elements in order to avoid
   * counting the ghost data more than once. To allow writing to ghost
   * elements again, call zero_out_ghost_values().
   *
   * @see
   * @ref GlossGhostedVector "vectors with ghost elements"
   */
  void
  update_ghost_values() const;

  /**
   * Initiates communication for the @p compress() function with non-
   * blocking communication. This function does not wait for the transfer
   * to finish, in order to allow for other computations during the time
   * it takes until all data arrives.
   *
   * Before the data is actually exchanged, the function must be followed
   * by a call to @p compress_finish().
   *
   * In case this function is called for more than one vector before @p
   * compress_finish() is invoked, it is mandatory to specify a unique
   * communication channel to each such call, in order to avoid several
   * messages with the same ID that will corrupt this operation. Any
   * communication channel less than 100 is a valid value (in particular,
   * the range $[100, 200)$ is reserved for
   * LinearAlgebra::distributed::BlockVector).
   */
  void
  compress_start(const unsigned int      communication_channel = 0,
                 VectorOperation::values operation = VectorOperation::add);

  /**
   * For all requests that have been initiated in compress_start, wait for
   * the communication to finish. Once it is finished, add or set the data
   * (depending on the flag operation) to the respective positions in the
   * owning processor, and clear the contents in the ghost data fields.
   * The meaning of this argument is the same as in compress().
   *
   * This function should be called exactly once per vector after calling
   * compress_start, otherwise the result is undefined. In particular, it
   * is not well-defined to call compress_start on the same vector again
   * before compress_finished has been called. However, there is no
   * warning to prevent this situation.
   *
   * Must follow a call to the @p compress_start function.
   *
   * When the MemorySpaceType is CUDA and MPI is not CUDA-aware, data changed on
   * the device after the call to compress_start will be lost.
   */
  void
  compress_finish(VectorOperation::values operation);

  /**
   * Initiates communication for the @p update_ghost_values() function
   * with non-blocking communication. This function does not wait for the
   * transfer to finish, in order to allow for other computations during
   * the time it takes until all data arrives.
   *
   * Before the data is actually exchanged, the function must be followed
   * by a call to @p update_ghost_values_finish().
   *
   * In case this function is called for more than one vector before @p
   * update_ghost_values_finish() is invoked, it is mandatory to specify a
   * unique communication channel to each such call, in order to avoid
   * several messages with the same ID that will corrupt this operation.
   * Any communication channel less than 100 is a valid value (in
   * particular, the range $[100, 200)$ is reserved for
   * LinearAlgebra::distributed::BlockVector).
   */
  void
  update_ghost_values_start(const unsigned int communication_channel = 0) const;


  /**
   * For all requests that have been started in update_ghost_values_start,
   * wait for the communication to finish.
   *
   * Must follow a call to the @p update_ghost_values_start function
   * before reading data from ghost indices.
   */
  void
  update_ghost_values_finish() const;

  /**
   * This method zeros the entries on ghost dofs, but does not touch
   * locally owned DoFs.
   *
   * After calling this method, read access to ghost elements of the
   * vector is forbidden and an exception is thrown. Only write access to
   * ghost elements is allowed in this state.
   *
   * @deprecated Use zero_out_ghost_values() instead.
   */
  DEAL_II_DEPRECATED_EARLY void
  zero_out_ghosts() const;

  /**
   * This method zeros the entries on ghost dofs, but does not touch
   * locally owned DoFs.
   *
   * After calling this method, read access to ghost elements of the
   * vector is forbidden and an exception is thrown. Only write access to
   * ghost elements is allowed in this state.
   */
  void
  zero_out_ghost_values() const;

  /**
   * Return whether the vector currently is in a state where ghost values
   * can be read or not. This is the same functionality as other parallel
   * vectors have. If this method returns false, this only means that
   * read-access to ghost elements is prohibited whereas write access is
   * still possible (to those entries specified as ghosts during
   * initialization), not that there are no ghost elements at all.
   *
   * @see
   * @ref GlossGhostedVector "vectors with ghost elements"
   */
  bool
  has_ghost_elements() const;

  /**
   * This method copies the data in the locally owned range from another
   * distributed vector @p src into the calling vector. As opposed to
   * operator= that also includes ghost entries, this operation ignores
   * the ghost range. The only prerequisite is that the local range on the
   * calling vector and the given vector @p src are the same on all
   * processors. It is explicitly allowed that the two vectors have
   * different ghost elements that might or might not be related to each
   * other.
   *
   * Since no data exchange is performed, make sure that neither @p src
   * nor the calling vector have pending communications in order to obtain
   * correct results.
   */
  template <typename Number2>
  void
  copy_locally_owned_data_from(
    const GhostedVector<Number2, MemorySpaceType> &src);

  /**
   * Import all the elements present in the distributed vector @p src.
   * VectorOperation::values @p operation is used to decide if the elements
   * in @p V should be added to the current vector or replace the current
   * elements. The main purpose of this function is to get data from one
   * memory space, e.g. CUDA, to the other, e.g. the Host.
   *
   * @note The partitioners of the two distributed vectors need to be the
   * same as no MPI communication is performed.
   */
  template <typename MemorySpaceType2>
  void
  import(const GhostedVector<Number, MemorySpaceType2> &src,
         VectorOperation::values                        operation);

  //@}

  /**
   * @name 3: Implementation of VectorSpaceVector
   */
  //@{

  /**
   * Return the global size of the vector, equal to the sum of the number of
   * locally owned indices among all processors.
   */
  virtual size_type
  size() const;

  /**
   * Return an index set that describes which elements of this vector are
   * owned by the current processor. As a consequence, the index sets
   * returned on different processors if this is a distributed vector will
   * form disjoint sets that add up to the complete index set. Obviously, if
   * a vector is created on only one processor, then the result would
   * satisfy
   * @code
   *  vec.locally_owned_elements() == complete_index_set(vec.size())
   * @endcode
   */
  virtual dealii::IndexSet
  locally_owned_elements() const;

  /**
   * Print the vector to the output stream @p out.
   */
  virtual void
  print(std::ostream &     out,
        const unsigned int precision  = 3,
        const bool         scientific = true,
        const bool         across     = true) const;

  /**
   * Return the memory consumption of this class in bytes.
   */
  virtual std::size_t
  memory_consumption() const;
  //@}

  /**
   * @name 5: Entry access and local data representation
   */
  //@{

  /**
   * Return the local size of the vector, i.e., the number of indices
   * owned locally.
   */
  size_type
  locally_owned_size() const;

  /**
   * Return true if the given global index is in the local range of this
   * processor.
   */
  bool
  in_local_range(const size_type global_index) const;

  /**
   * Make the @p Vector class a bit like the <tt>vector<></tt> class of
   * the C++ standard library by returning iterators to the start and end
   * of the <i>locally owned</i> elements of this vector.
   *
   * It holds that end() - begin() == locally_owned_size().
   *
   * @note For the CUDA memory space, the iterator points to memory on the
   * device.
   */
  iterator
  begin();

  /**
   * Return constant iterator to the start of the locally owned elements
   * of the vector.
   *
   * @note For the CUDA memory space, the iterator points to memory on the
   * device.
   */
  const_iterator
  begin() const;

  /**
   * Return an iterator pointing to the element past the end of the array
   * of locally owned entries.
   *
   * @note For the CUDA memory space, the iterator points to memory on the
   * device.
   */
  iterator
  end();

  /**
   * Return a constant iterator pointing to the element past the end of
   * the array of the locally owned entries.
   *
   * @note For the CUDA memory space, the iterator points to memory on the
   * device.
   */
  const_iterator
  end() const;

  /**
   * Read access to the data in the position corresponding to @p
   * global_index. The index must be either in the local range of the
   * vector or be specified as a ghost index at construction.
   *
   * Performance: <tt>O(1)</tt> for locally owned elements that represent
   * a contiguous range and <tt>O(log(n<sub>ranges</sub>))</tt> for ghost
   * elements (quite fast, but slower than local_element()).
   */
  Number
  operator()(const size_type global_index) const;

  /**
   * Read and write access to the data in the position corresponding to @p
   * global_index. The index must be either in the local range of the
   * vector or be specified as a ghost index at construction.
   *
   * Performance: <tt>O(1)</tt> for locally owned elements that represent
   * a contiguous range and <tt>O(log(n<sub>ranges</sub>))</tt> for ghost
   * elements (quite fast, but slower than local_element()).
   */
  Number &
  operator()(const size_type global_index);

  /**
   * Read access to the data in the position corresponding to @p
   * global_index. The index must be either in the local range of the
   * vector or be specified as a ghost index at construction.
   *
   * This function does the same thing as operator().
   */
  Number operator[](const size_type global_index) const;

  /**
   * Read and write access to the data in the position corresponding to @p
   * global_index. The index must be either in the local range of the
   * vector or be specified as a ghost index at construction.
   *
   * This function does the same thing as operator().
   */
  Number &operator[](const size_type global_index);

  /**
   * Read access to the data field specified by @p local_index. Locally
   * owned indices can be accessed with indices
   * <code>[0,locally_owned_size)</code>, and ghost indices with indices
   * <code>[locally_owned_size,locally_owned_size+ n_ghost_entries]</code>.
   *
   * Performance: Direct array access (fast).
   */
  Number
  local_element(const size_type local_index) const;

  /**
   * Read and write access to the data field specified by @p local_index.
   * Locally owned indices can be accessed with indices
   * <code>[0,locally_owned_size())</code>, and ghost indices with indices
   * <code>[locally_owned_size(), locally_owned_size()+n_ghosts]</code>.
   *
   * Performance: Direct array access (fast).
   */
  Number &
  local_element(const size_type local_index);

  /**
   * Return the pointer to the underlying raw array.
   *
   * @note For the CUDA memory space, the pointer points to memory on the
   * device.
   */
  Number *
  get_values() const;

  /**
   * Instead of getting individual elements of a vector via operator(),
   * this function allows getting a whole set of elements at once. The
   * indices of the elements to be read are stated in the first argument,
   * the corresponding values are returned in the second.
   *
   * If the current vector is called @p v, then this function is the equivalent
   * to the code
   * @code
   *   for (unsigned int i=0; i<indices.size(); ++i)
   *     values[i] = v[indices[i]];
   * @endcode
   *
   * @pre The sizes of the @p indices and @p values arrays must be identical.
   *
   * @note This function is not implemented for CUDA memory space.
   */
  template <typename OtherNumber>
  void
  extract_subvector_to(const std::vector<size_type> &indices,
                       std::vector<OtherNumber> &    values) const;

  /**
   * Instead of getting individual elements of a vector via operator(),
   * this function allows getting a whole set of elements at once. In
   * contrast to the previous function, this function obtains the
   * indices of the elements by dereferencing all elements of the iterator
   * range provided by the first two arguments, and puts the vector
   * values into memory locations obtained by dereferencing a range
   * of iterators starting at the location pointed to by the third
   * argument.
   *
   * If the current vector is called @p v, then this function is the equivalent
   * to the code
   * @code
   *   ForwardIterator indices_p = indices_begin;
   *   OutputIterator  values_p  = values_begin;
   *   while (indices_p != indices_end)
   *   {
   *     *values_p = v[*indices_p];
   *     ++indices_p;
   *     ++values_p;
   *   }
   * @endcode
   *
   * @pre It must be possible to write into as many memory locations
   *   starting at @p values_begin as there are iterators between
   *   @p indices_begin and @p indices_end.
   */
  template <typename ForwardIterator, typename OutputIterator>
  void
  extract_subvector_to(ForwardIterator       indices_begin,
                       const ForwardIterator indices_end,
                       OutputIterator        values_begin) const;

  /**
   * @name 6: Mixed stuff
   */
  //@{

  /**
   * Return a reference to the MPI communicator object in use with this
   * vector.
   */
  const MPI_Comm &
  get_mpi_communicator() const;

  /**
   * Return the MPI partitioner that describes the parallel layout of the
   * vector. This object can be used to initialize another vector with the
   * respective reinit() call, for additional queries regarding the
   * parallel communication, or the compatibility of partitioners.
   */
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_partitioner() const;

  /**
   * Check whether the given partitioner is compatible with the
   * partitioner used for this vector. Two partitioners are compatible if
   * they have the same local size and the same ghost indices. They do not
   * necessarily need to be the same data field of the shared pointer.
   * This is a local operation only, i.e., if only some processors decide
   * that the partitioning is not compatible, only these processors will
   * return @p false, whereas the other processors will return @p true.
   */
  bool
  partitioners_are_compatible(const Utilities::MPI::Partitioner &part) const;

  /**
   * Check whether the given partitioner is compatible with the
   * partitioner used for this vector. Two partitioners are compatible if
   * they have the same local size and the same ghost indices. They do not
   * necessarily need to be the same data field. As opposed to
   * partitioners_are_compatible(), this method checks for compatibility
   * among all processors and the method only returns @p true if the
   * partitioner is the same on all processors.
   *
   * This method performs global communication, so make sure to use it
   * only in a context where all processors call it the same number of
   * times.
   */
  bool
  partitioners_are_globally_compatible(
    const Utilities::MPI::Partitioner &part) const;

  /**
   * Change the ghost state of this vector to @p ghosted.
   */
  void
  set_ghost_state(const bool ghosted) const;

  /**
   * Get pointers to the beginning of the values of the other
   * processes of the same shared-memory domain.
   */
  const std::vector<ArrayView<const Number>> &
  shared_vector_data() const;

  //@}

  /**
   * Attempt to perform an operation between two incompatible vector types.
   *
   * @ingroup Exceptions
   */
  DeclException0(ExcVectorTypeNotCompatible);

  /**
   * Attempt to perform an operation not implemented on the device.
   *
   * @ingroup Exceptions
   */
  DeclException0(ExcNotAllowedForCuda);

  /**
   * Exception
   */
  DeclException3(ExcNonMatchingElements,
                 Number,
                 Number,
                 unsigned int,
                 << "Called compress(VectorOperation::insert), but"
                 << " the element received from a remote processor, value "
                 << std::setprecision(16) << arg1
                 << ", does not match with the value " << std::setprecision(16)
                 << arg2 << " on the owner processor " << arg3);

  /**
   * Exception
   */
  DeclException4(
    ExcAccessToNonLocalElement,
    size_type,
    size_type,
    size_type,
    size_type,
    << "You tried to access element " << arg1
    << " of a distributed vector, but this element is not "
    << "stored on the current processor. Note: The range of "
    << "locally owned elements is [" << arg2 << "," << arg3
    << "], and there are " << arg4 << " ghost elements "
    << "that this vector can access."
    << "\n\n"
    << "A common source for this kind of problem is that you "
    << "are passing a 'fully distributed' vector into a function "
    << "that needs read access to vector elements that correspond "
    << "to degrees of freedom on ghost cells (or at least to "
    << "'locally active' degrees of freedom that are not also "
    << "'locally owned'). You need to pass a vector that has these "
    << "elements as ghost entries.");

private:
  /**
   * Shared pointer to store the parallel partitioning information. This
   * information can be shared between several vectors that have the same
   * partitioning.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  /**
   * The size that is currently allocated in the val array.
   */
  size_type allocated_size;

  /**
   * Underlying data structure storing the local elements of this vector.
   */
  mutable MemorySpace::MemorySpaceData<Number, MemorySpaceType> data;

  /**
   * For parallel loops with TBB, this member variable stores the affinity
   * information of loops.
   */
  mutable std::shared_ptr<parallel::internal::TBBPartitioner>
    thread_loop_partitioner;

  /**
   * Temporary storage that holds the data that is sent to this processor
   * in compress() or sent from this processor in update_ghost_values().
   */
  mutable MemorySpace::MemorySpaceData<Number, MemorySpaceType> import_data;

  /**
   * Stores whether the vector currently allows for reading ghost elements
   * or not. Note that this is to ensure consistent ghost data and does
   * not indicate whether the vector actually can store ghost elements. In
   * particular, when assembling a vector we do not allow reading
   * elements, only writing them.
   */
  mutable bool vector_is_ghosted;

#ifdef DEAL_II_WITH_MPI
  /**
   * A vector that collects all requests from compress() operations.
   * This class uses persistent MPI communicators, i.e., the communication
   * channels are stored during successive calls to a given function. This
   * reduces the overhead involved with setting up the MPI machinery, but
   * it does not remove the need for a receive operation to be posted
   * before the data can actually be sent.
   */
  std::vector<MPI_Request> compress_requests;

  /**
   * A vector that collects all requests from update_ghost_values()
   * operations. This class uses persistent MPI communicators.
   */
  mutable std::vector<MPI_Request> update_ghost_values_requests;
#endif

  /**
   * A lock that makes sure that the compress() and update_ghost_values()
   * functions give reasonable results also when used
   * with several threads.
   */
  mutable std::mutex mutex;

  /**
   * Communicator to be used for the shared-memory domain.
   */
  MPI_Comm comm_sm;

  /**
   * A helper function that clears the compress_requests and
   * update_ghost_values_requests field. Used in reinit() functions.
   */
  void
  clear_mpi_requests();

  /**
   * A helper function that is used to resize the val array.
   */
  void
  resize_val(const size_type new_allocated_size,
             const MPI_Comm &comm_sm = MPI_COMM_SELF);

  // Make all other vector types friends.
  template <typename Number2, typename MemorySpaceType2>
  friend class GhostedVector;
};
/*@}*/


/*-------------------- Inline functions ---------------------------------*/

#ifndef DOXYGEN

namespace internal
{
  template <typename Number, typename MemorySpaceType>
  struct Policy
  {
    static inline typename GhostedVector<Number, MemorySpaceType>::iterator
    begin(MemorySpace::MemorySpaceData<Number, MemorySpaceType> &)
    {
      return nullptr;
    }

    static inline
      typename GhostedVector<Number, MemorySpaceType>::const_iterator
      begin(const MemorySpace::MemorySpaceData<Number, MemorySpaceType> &)
    {
      return nullptr;
    }

    static inline Number *
    get_values(MemorySpace::MemorySpaceData<Number, MemorySpaceType> &)
    {
      return nullptr;
    }
  };



  template <typename Number>
  struct Policy<Number, MemorySpace::Host>
  {
    static inline typename GhostedVector<Number, MemorySpace::Host>::iterator
    begin(MemorySpace::MemorySpaceData<Number, MemorySpace::Host> &data)
    {
      return data.values.get();
    }

    static inline
      typename GhostedVector<Number, MemorySpace::Host>::const_iterator
      begin(const MemorySpace::MemorySpaceData<Number, MemorySpace::Host> &data)
    {
      return data.values.get();
    }

    static inline Number *
    get_values(MemorySpace::MemorySpaceData<Number, MemorySpace::Host> &data)
    {
      return data.values.get();
    }
  };



  template <typename Number>
  struct Policy<Number, MemorySpace::CUDA>
  {
    static inline typename GhostedVector<Number, MemorySpace::CUDA>::iterator
    begin(MemorySpace::MemorySpaceData<Number, MemorySpace::CUDA> &data)
    {
      return data.values_dev.get();
    }

    static inline
      typename GhostedVector<Number, MemorySpace::CUDA>::const_iterator
      begin(const MemorySpace::MemorySpaceData<Number, MemorySpace::CUDA> &data)
    {
      return data.values_dev.get();
    }

    static inline Number *
    get_values(MemorySpace::MemorySpaceData<Number, MemorySpace::CUDA> &data)
    {
      return data.values_dev.get();
    }
  };
} // namespace internal


template <typename Number, typename MemorySpaceType>
inline bool
GhostedVector<Number, MemorySpaceType>::has_ghost_elements() const
{
  return vector_is_ghosted;
}



template <typename Number, typename MemorySpaceType>
inline typename GhostedVector<Number, MemorySpaceType>::size_type
GhostedVector<Number, MemorySpaceType>::size() const
{
  return partitioner->size();
}



template <typename Number, typename MemorySpaceType>
inline typename GhostedVector<Number, MemorySpaceType>::size_type
GhostedVector<Number, MemorySpaceType>::locally_owned_size() const
{
  return partitioner->locally_owned_size();
}



template <typename Number, typename MemorySpaceType>
inline bool
GhostedVector<Number, MemorySpaceType>::in_local_range(
  const size_type global_index) const
{
  return partitioner->in_local_range(global_index);
}



template <typename Number, typename MemorySpaceType>
inline IndexSet
GhostedVector<Number, MemorySpaceType>::locally_owned_elements() const
{
  IndexSet is(size());

  is.add_range(partitioner->local_range().first,
               partitioner->local_range().second);

  return is;
}



template <typename Number, typename MemorySpaceType>
inline typename GhostedVector<Number, MemorySpaceType>::iterator
GhostedVector<Number, MemorySpaceType>::begin()
{
  return internal::Policy<Number, MemorySpaceType>::begin(data);
}



template <typename Number, typename MemorySpaceType>
inline typename GhostedVector<Number, MemorySpaceType>::const_iterator
GhostedVector<Number, MemorySpaceType>::begin() const
{
  return internal::Policy<Number, MemorySpaceType>::begin(data);
}



template <typename Number, typename MemorySpaceType>
inline typename GhostedVector<Number, MemorySpaceType>::iterator
GhostedVector<Number, MemorySpaceType>::end()
{
  return internal::Policy<Number, MemorySpaceType>::begin(data) +
         partitioner->locally_owned_size();
}



template <typename Number, typename MemorySpaceType>
inline typename GhostedVector<Number, MemorySpaceType>::const_iterator
GhostedVector<Number, MemorySpaceType>::end() const
{
  return internal::Policy<Number, MemorySpaceType>::begin(data) +
         partitioner->locally_owned_size();
}



template <typename Number, typename MemorySpaceType>
const std::vector<ArrayView<const Number>> &
GhostedVector<Number, MemorySpaceType>::shared_vector_data() const
{
  return data.values_sm;
}



template <typename Number, typename MemorySpaceType>
inline Number
GhostedVector<Number, MemorySpaceType>::
operator()(const size_type global_index) const
{
  Assert((std::is_same<MemorySpaceType, MemorySpace::Host>::value),
         ExcMessage(
           "This function is only implemented for the Host memory space"));
  Assert(partitioner->in_local_range(global_index) ||
           partitioner->ghost_indices().is_element(global_index),
         ExcAccessToNonLocalElement(global_index,
                                    partitioner->local_range().first,
                                    partitioner->local_range().second - 1,
                                    partitioner->ghost_indices().n_elements()));
  // do not allow reading a vector which is not in ghost mode
  Assert(partitioner->in_local_range(global_index) || vector_is_ghosted == true,
         ExcMessage("You tried to read a ghost element of this vector, "
                    "but it has not imported its ghost values."));
  return data.values[partitioner->global_to_local(global_index)];
}



template <typename Number, typename MemorySpaceType>
inline Number &
GhostedVector<Number, MemorySpaceType>::operator()(const size_type global_index)
{
  Assert((std::is_same<MemorySpaceType, MemorySpace::Host>::value),
         ExcMessage(
           "This function is only implemented for the Host memory space"));
  Assert(partitioner->in_local_range(global_index) ||
           partitioner->ghost_indices().is_element(global_index),
         ExcAccessToNonLocalElement(global_index,
                                    partitioner->local_range().first,
                                    partitioner->local_range().second - 1,
                                    partitioner->ghost_indices().n_elements()));
  // we would like to prevent reading ghosts from a vector that does not
  // have them imported, but this is not possible because we might be in a
  // part of the code where the vector has enabled ghosts but is non-const
  // (then, the compiler picks this method according to the C++ rule book
  // even if a human would pick the const method when this subsequent use
  // is just a read)
  return data.values[partitioner->global_to_local(global_index)];
}



template <typename Number, typename MemorySpaceType>
inline Number GhostedVector<Number, MemorySpaceType>::
              operator[](const size_type global_index) const
{
  return operator()(global_index);
}



template <typename Number, typename MemorySpaceType>
inline Number &GhostedVector<Number, MemorySpaceType>::
               operator[](const size_type global_index)
{
  return operator()(global_index);
}



template <typename Number, typename MemorySpaceType>
inline Number
GhostedVector<Number, MemorySpaceType>::local_element(
  const size_type local_index) const
{
  Assert((std::is_same<MemorySpaceType, MemorySpace::Host>::value),
         ExcMessage(
           "This function is only implemented for the Host memory space"));
  AssertIndexRange(local_index,
                   partitioner->locally_owned_size() +
                     partitioner->n_ghost_indices());
  // do not allow reading a vector which is not in ghost mode
  Assert(local_index < locally_owned_size() || vector_is_ghosted == true,
         ExcMessage("You tried to read a ghost element of this vector, "
                    "but it has not imported its ghost values."));

  return data.values[local_index];
}



template <typename Number, typename MemorySpaceType>
inline Number &
GhostedVector<Number, MemorySpaceType>::local_element(
  const size_type local_index)
{
  Assert((std::is_same<MemorySpaceType, MemorySpace::Host>::value),
         ExcMessage(
           "This function is only implemented for the Host memory space"));

  AssertIndexRange(local_index,
                   partitioner->locally_owned_size() +
                     partitioner->n_ghost_indices());

  return data.values[local_index];
}



template <typename Number, typename MemorySpaceType>
inline Number *
GhostedVector<Number, MemorySpaceType>::get_values() const
{
  return internal::Policy<Number, MemorySpaceType>::get_values(data);
}



template <typename Number, typename MemorySpaceType>
template <typename OtherNumber>
inline void
GhostedVector<Number, MemorySpaceType>::extract_subvector_to(
  const std::vector<size_type> &indices,
  std::vector<OtherNumber> &    values) const
{
  for (size_type i = 0; i < indices.size(); ++i)
    values[i] = operator()(indices[i]);
}



template <typename Number, typename MemorySpaceType>
template <typename ForwardIterator, typename OutputIterator>
inline void
GhostedVector<Number, MemorySpaceType>::extract_subvector_to(
  ForwardIterator       indices_begin,
  const ForwardIterator indices_end,
  OutputIterator        values_begin) const
{
  while (indices_begin != indices_end)
    {
      *values_begin = operator()(*indices_begin);
      indices_begin++;
      values_begin++;
    }
}



template <typename Number, typename MemorySpaceType>
inline const MPI_Comm &
GhostedVector<Number, MemorySpaceType>::get_mpi_communicator() const
{
  return partitioner->get_mpi_communicator();
}



template <typename Number, typename MemorySpaceType>
inline const std::shared_ptr<const Utilities::MPI::Partitioner> &
GhostedVector<Number, MemorySpaceType>::get_partitioner() const
{
  return partitioner;
}



template <typename Number, typename MemorySpaceType>
inline void
GhostedVector<Number, MemorySpaceType>::set_ghost_state(
  const bool ghosted) const
{
  vector_is_ghosted = ghosted;
}

#endif


/**
 * Global function @p swap which overloads the default implementation of the
 * C++ standard library which uses a temporary object. The function simply
 * exchanges the data of the two vectors.
 *
 * @relatesalso Vector
 */
template <typename Number, typename MemorySpaceType>
inline void
swap(GhostedVector<Number, MemorySpaceType> &u,
     GhostedVector<Number, MemorySpaceType> &v)
{
  u.swap(v);
}


/**
 * Declare dealii::LinearAlgebra::Vector as distributed vector.
 */
template <typename Number, typename MemorySpaceType>
struct is_serial_vector<GhostedVector<Number, MemorySpaceType>>
  : std::false_type
{};


DEAL_II_NAMESPACE_CLOSE

#endif
