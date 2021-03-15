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

#ifndef dealii_ghosted_vector_templates_h
#define dealii_ghosted_vector_templates_h


#include <deal.II/base/config.h>

#include <deal.II/base/cuda.h>
#include <deal.II/base/cuda_size.h>
#include <deal.II/base/ghosted_vector.h>

#include <deal.II/lac/vector_operations_internal.h>

#include <memory>


DEAL_II_NAMESPACE_OPEN


namespace internal
{
  // Resize the underlying array on the host or on the device
  template <typename Number, typename MemorySpaceType>
  struct la_parallel_vector_templates_functions
  {
    static_assert(std::is_same<MemorySpaceType, MemorySpace::Host>::value ||
                    std::is_same<MemorySpaceType, MemorySpace::CUDA>::value,
                  "MemorySpace should be Host or CUDA");

    static void
    resize_val(const types::global_dof_index /*new_alloc_size*/,
               types::global_dof_index & /*allocated_size*/,
               ::dealii::MemorySpace::MemorySpaceData<Number, MemorySpaceType>
                 & /*data*/,
               const MPI_Comm & /*comm_sm*/)
    {}
  };

  template <typename Number>
  struct la_parallel_vector_templates_functions<Number,
                                                ::dealii::MemorySpace::Host>
  {
    using size_type = types::global_dof_index;

    static void
    resize_val(
      const types::global_dof_index new_alloc_size,
      types::global_dof_index &     allocated_size,
      ::dealii::MemorySpace::MemorySpaceData<Number,
                                             ::dealii::MemorySpace::Host> &data,
      const MPI_Comm &comm_shared)
    {
      if (comm_shared == MPI_COMM_SELF)
        {
          Number *new_val;
          Utilities::System::posix_memalign(reinterpret_cast<void **>(&new_val),
                                            64,
                                            sizeof(Number) * new_alloc_size);
          data.values = {new_val, [](Number *data) { std::free(data); }};

          allocated_size = new_alloc_size;

          data.values_sm = {
            ArrayView<const Number>(data.values.get(), new_alloc_size)};
        }
      else
        {
#ifdef DEAL_II_WITH_MPI
#  if DEAL_II_MPI_VERSION_GTE(3, 0)
          allocated_size = new_alloc_size;

          const unsigned int size_sm =
            Utilities::MPI::n_mpi_processes(comm_shared);
          const unsigned int rank_sm =
            Utilities::MPI::this_mpi_process(comm_shared);

          MPI_Win mpi_window;
          Number *data_this;


          std::vector<Number *> others(size_sm);

          MPI_Info info;
          MPI_Info_create(&info);

          MPI_Info_set(info, "alloc_shared_noncontig", "true");

          const std::size_t align_by = 64;

          std::size_t s = ((new_alloc_size * sizeof(Number) + align_by - 1) /
                           sizeof(Number)) *
                          sizeof(Number);

          auto ierr = MPI_Win_allocate_shared(
            s, sizeof(Number), info, comm_shared, &data_this, &mpi_window);
          AssertThrowMPI(ierr);

          for (unsigned int i = 0; i < size_sm; i++)
            {
              int        disp_unit;
              MPI_Aint   ssize;
              const auto ierr = MPI_Win_shared_query(
                mpi_window, i, &ssize, &disp_unit, &others[i]);
              AssertThrowMPI(ierr);
            }

          Number *ptr_unaligned = others[rank_sm];
          Number *ptr_aligned   = ptr_unaligned;

          AssertThrow(std::align(align_by,
                                 new_alloc_size * sizeof(Number),
                                 reinterpret_cast<void *&>(ptr_aligned),
                                 s) != nullptr,
                      ExcNotImplemented());

          unsigned int              n_align_local = ptr_aligned - ptr_unaligned;
          std::vector<unsigned int> n_align_sm(size_sm);

          ierr = MPI_Allgather(&n_align_local,
                               1,
                               MPI_UNSIGNED,
                               n_align_sm.data(),
                               1,
                               MPI_UNSIGNED,
                               comm_shared);
          AssertThrowMPI(ierr);

          for (unsigned int i = 0; i < size_sm; i++)
            others[i] += n_align_sm[i];

          std::vector<unsigned int> new_alloc_sizes(size_sm);

          ierr = MPI_Allgather(&new_alloc_size,
                               1,
                               MPI_UNSIGNED,
                               new_alloc_sizes.data(),
                               1,
                               MPI_UNSIGNED,
                               comm_shared);
          AssertThrowMPI(ierr);

          data.values_sm.resize(size_sm);
          for (unsigned int i = 0; i < size_sm; i++)
            data.values_sm[i] =
              ArrayView<const Number>(others[i], new_alloc_sizes[i]);

          data.values = {ptr_aligned, [mpi_window](Number *) mutable {
                           // note: we are creating here a copy of the
                           // window other approaches led to segmentation
                           // faults
                           const auto ierr = MPI_Win_free(&mpi_window);
                           AssertThrowMPI(ierr);
                         }};
#  else
          AssertThrow(
            false, ExcMessage("Sorry, this feature requires MPI 3.0 support"));
#  endif
#else
          Assert(false, ExcInternalError());
#endif
        }
    }
  };

#ifdef DEAL_II_COMPILER_CUDA_AWARE
  template <typename Number>
  struct la_parallel_vector_templates_functions<Number,
                                                ::dealii::MemorySpace::CUDA>
  {
    using size_type = types::global_dof_index;

    static void
    resize_val(
      const types::global_dof_index new_alloc_size,
      types::global_dof_index &     allocated_size,
      ::dealii::MemorySpace::MemorySpaceData<Number,
                                             ::dealii::MemorySpace::CUDA> &data,
      const MPI_Comm &comm_sm)
    {
      (void)comm_sm;

      static_assert(std::is_same<Number, float>::value ||
                      std::is_same<Number, double>::value,
                    "Number should be float or double for CUDA memory space");

      if (new_alloc_size > allocated_size)
        {
          Assert(((allocated_size > 0 && data.values_dev != nullptr) ||
                  data.values_dev == nullptr),
                 ExcInternalError());

          Number *new_val_dev;
          Utilities::CUDA::malloc(new_val_dev, new_alloc_size);
          data.values_dev.reset(new_val_dev);

          allocated_size = new_alloc_size;
        }
      else if (new_alloc_size == 0)
        {
          data.values_dev.reset();
          allocated_size = 0;
        }
    }
  };
#endif
} // namespace internal


template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::clear_mpi_requests()
{
#ifdef DEAL_II_WITH_MPI
  for (auto &compress_request : compress_requests)
    {
      const int ierr = MPI_Request_free(&compress_request);
      AssertThrowMPI(ierr);
    }
  compress_requests.clear();
  for (auto &update_ghost_values_request : update_ghost_values_requests)
    {
      const int ierr = MPI_Request_free(&update_ghost_values_request);
      AssertThrowMPI(ierr);
    }
  update_ghost_values_requests.clear();
#endif
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::resize_val(
  const size_type new_alloc_size,
  const MPI_Comm &comm_sm)
{
  // TODO: Replace this with the actual resize_val function
  internal::la_parallel_vector_templates_functions<Number, MemorySpaceType>::
    resize_val(new_alloc_size, allocated_size, data, comm_sm);

  thread_loop_partitioner =
    std::make_shared<parallel::internal::TBBPartitioner>();
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::reinit(const size_type size,
                                               const bool omit_zeroing_entries)
{
  clear_mpi_requests();

  // check whether we need to reallocate
  resize_val(size, comm_sm);

  // delete previous content in import data
  import_data.values.reset();
  import_data.values_dev.reset();

  // set partitioner to serial version
  partitioner = std::make_shared<Utilities::MPI::Partitioner>(size);

  // set entries to zero if so requested
  if (omit_zeroing_entries == false)
    {
      const size_type this_size = locally_owned_size();
      if (this_size > 0)
        {
          internal::VectorOperations::
            functions<Number, Number, MemorySpaceType>::set(
              thread_loop_partitioner, this_size, Number(), data);
        }
      zero_out_ghost_values();
    }
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::reinit(
  const types::global_dof_index local_size,
  const types::global_dof_index ghost_size,
  const MPI_Comm &              comm,
  const MPI_Comm &              comm_sm)
{
  clear_mpi_requests();

  this->comm_sm = comm_sm;

  // check whether we need to reallocate
  resize_val(local_size + ghost_size, comm_sm);

  // delete previous content in import data
  import_data.values.reset();
  import_data.values_dev.reset();

  // create partitioner
  partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(local_size, ghost_size, comm);

  // initialize entries to zero
  const size_type this_size = locally_owned_size();
  if (this_size > 0)
    {
      internal::VectorOperations::functions<Number, Number, MemorySpaceType>::
        set(thread_loop_partitioner, this_size, Number(), data);
    }
  zero_out_ghost_values();
}



template <typename Number, typename MemorySpaceType>
template <typename Number2>
void
GhostedVector<Number, MemorySpaceType>::reinit(
  const GhostedVector<Number2, MemorySpaceType> &v,
  const bool                                     omit_zeroing_entries)
{
  clear_mpi_requests();
  Assert(v.partitioner.get() != nullptr, ExcNotInitialized());

  this->comm_sm = v.comm_sm;

  // check whether the partitioners are
  // different (check only if the are allocated
  // differently, not if the actual data is
  // different)
  if (partitioner.get() != v.partitioner.get())
    {
      partitioner = v.partitioner;
      const size_type new_allocated_size =
        partitioner->locally_owned_size() + partitioner->n_ghost_indices();
      resize_val(new_allocated_size, this->comm_sm);
    }

  // set entries to zero if so requested
  if (omit_zeroing_entries == false)
    {
      const size_type this_size = locally_owned_size();
      if (this_size > 0)
        {
          internal::VectorOperations::
            functions<Number, Number, MemorySpaceType>::set(
              thread_loop_partitioner, this_size, Number(), data);
        }
      zero_out_ghost_values();
    }

  // do not reallocate import_data directly, but only upon request. It
  // is only used as temporary storage for compress() and
  // update_ghost_values, and we might have vectors where we never
  // call these methods and hence do not need to have the storage.
  import_data.values.reset();
  import_data.values_dev.reset();

  thread_loop_partitioner = v.thread_loop_partitioner;
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::reinit(
  const IndexSet &locally_owned_indices,
  const IndexSet &ghost_indices,
  const MPI_Comm &communicator)
{
  // set up parallel partitioner with index sets and communicator
  reinit(std::make_shared<Utilities::MPI::Partitioner>(locally_owned_indices,
                                                       ghost_indices,
                                                       communicator));
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::reinit(
  const IndexSet &locally_owned_indices,
  const MPI_Comm &communicator)
{
  // set up parallel partitioner with index sets and communicator
  reinit(std::make_shared<Utilities::MPI::Partitioner>(locally_owned_indices,
                                                       communicator));
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::reinit(
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner_in,
  const MPI_Comm &                                          comm_sm)
{
  clear_mpi_requests();
  partitioner = partitioner_in;

  this->comm_sm = comm_sm;

  // set vector size and allocate memory
  const size_type new_allocated_size =
    partitioner->locally_owned_size() + partitioner->n_ghost_indices();
  resize_val(new_allocated_size, comm_sm);

  // initialize entries to zero
  const size_type this_size = locally_owned_size();
  if (this_size > 0)
    {
      internal::VectorOperations::functions<Number, Number, MemorySpaceType>::
        set(thread_loop_partitioner, this_size, Number(), data);
    }
  zero_out_ghost_values();


  // do not reallocate import_data directly, but only upon request. It
  // is only used as temporary storage for compress() and
  // update_ghost_values, and we might have vectors where we never
  // call these methods and hence do not need to have the storage.
  import_data.values.reset();
  import_data.values_dev.reset();

  vector_is_ghosted = false;
}



template <typename Number, typename MemorySpaceType>
GhostedVector<Number, MemorySpaceType>::GhostedVector()
  : partitioner(std::make_shared<Utilities::MPI::Partitioner>())
  , allocated_size(0)
  , comm_sm(MPI_COMM_SELF)
{
  reinit(0);
}



template <typename Number, typename MemorySpaceType>
GhostedVector<Number, MemorySpaceType>::GhostedVector(
  const GhostedVector<Number, MemorySpaceType> &v)
  : Subscriptor()
  , allocated_size(0)
  , vector_is_ghosted(false)
  , comm_sm(MPI_COMM_SELF)
{
  reinit(v, true);

  thread_loop_partitioner = v.thread_loop_partitioner;

  const size_type this_size = locally_owned_size();
  if (this_size > 0)
    {
      internal::VectorOperations::functions<Number, Number, MemorySpaceType>::
        copy(thread_loop_partitioner,
             partitioner->locally_owned_size(),
             v.data,
             data);
    }
}



template <typename Number, typename MemorySpaceType>
GhostedVector<Number, MemorySpaceType>::GhostedVector(
  const IndexSet &local_range,
  const IndexSet &ghost_indices,
  const MPI_Comm &communicator)
  : allocated_size(0)
  , vector_is_ghosted(false)
  , comm_sm(MPI_COMM_SELF)
{
  reinit(local_range, ghost_indices, communicator);
}



template <typename Number, typename MemorySpaceType>
GhostedVector<Number, MemorySpaceType>::GhostedVector(
  const IndexSet &local_range,
  const MPI_Comm &communicator)
  : allocated_size(0)
  , vector_is_ghosted(false)
  , comm_sm(MPI_COMM_SELF)
{
  reinit(local_range, communicator);
}



template <typename Number, typename MemorySpaceType>
GhostedVector<Number, MemorySpaceType>::GhostedVector(const size_type size)
  : allocated_size(0)
  , vector_is_ghosted(false)
  , comm_sm(MPI_COMM_SELF)
{
  reinit(size, false);
}



template <typename Number, typename MemorySpaceType>
GhostedVector<Number, MemorySpaceType>::GhostedVector(
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner)
  : allocated_size(0)
  , vector_is_ghosted(false)
  , comm_sm(MPI_COMM_SELF)
{
  reinit(partitioner);
}



template <typename Number, typename MemorySpaceType>
inline GhostedVector<Number, MemorySpaceType>::~GhostedVector()
{
  try
    {
      clear_mpi_requests();
    }
  catch (...)
    {}
}



template <typename Number, typename MemorySpaceType>
inline GhostedVector<Number, MemorySpaceType> &
GhostedVector<Number, MemorySpaceType>::
operator=(const GhostedVector<Number, MemorySpaceType> &c)
{
#ifdef _MSC_VER
  return this->operator=<Number>(c);
#else
  return this->template operator=<Number>(c);
#endif
}



template <typename Number, typename MemorySpaceType>
template <typename Number2>
inline GhostedVector<Number, MemorySpaceType> &
GhostedVector<Number, MemorySpaceType>::
operator=(const GhostedVector<Number2, MemorySpaceType> &c)
{
  Assert(c.partitioner.get() != nullptr, ExcNotInitialized());

  // we update ghost values whenever one of the input or output vector
  // already held ghost values or when we import data from a vector with
  // the same local range but different ghost layout
  bool must_update_ghost_values = c.vector_is_ghosted;

  this->comm_sm = c.comm_sm;

  // check whether the two vectors use the same parallel partitioner. if
  // not, check if all local ranges are the same (that way, we can
  // exchange data between different parallel layouts). One variant which
  // is included here and necessary for compatibility with the other
  // distributed vector classes (Trilinos, PETSc) is the case when vector
  // c does not have any ghosts (constructed without ghost elements given)
  // but the current vector does: In that case, we need to exchange data
  // also when none of the two vector had updated its ghost values before.
  if (partitioner.get() == nullptr)
    reinit(c, true);
  else if (partitioner.get() != c.partitioner.get())
    {
      // local ranges are also the same if both partitioners are empty
      // (even if they happen to define the empty range as [0,0) or [c,c)
      // for some c!=0 in a different way).
      int local_ranges_are_identical =
        (partitioner->local_range() == c.partitioner->local_range() ||
         (partitioner->local_range().second ==
            partitioner->local_range().first &&
          c.partitioner->local_range().second ==
            c.partitioner->local_range().first));
      if ((c.partitioner->n_mpi_processes() > 1 &&
           Utilities::MPI::min(local_ranges_are_identical,
                               c.partitioner->get_mpi_communicator()) == 0) ||
          !local_ranges_are_identical)
        reinit(c, true);
      else
        must_update_ghost_values |= vector_is_ghosted;

      must_update_ghost_values |=
        (c.partitioner->ghost_indices_initialized() == false &&
         partitioner->ghost_indices_initialized() == true);
    }
  else
    must_update_ghost_values |= vector_is_ghosted;

  thread_loop_partitioner = c.thread_loop_partitioner;

  const size_type this_size = partitioner->locally_owned_size();
  if (this_size > 0)
    {
      internal::VectorOperations::functions<Number, Number2, MemorySpaceType>::
        copy(thread_loop_partitioner, this_size, c.data, data);
    }

  if (must_update_ghost_values)
    update_ghost_values();
  else
    zero_out_ghost_values();
  return *this;
}



template <typename Number, typename MemorySpaceType>
template <typename Number2>
void
GhostedVector<Number, MemorySpaceType>::copy_locally_owned_data_from(
  const GhostedVector<Number2, MemorySpaceType> &src)
{
  AssertDimension(partitioner->locally_owned_size(),
                  src.partitioner->locally_owned_size());
  if (partitioner->locally_owned_size() > 0)
    {
      internal::VectorOperations::functions<Number, Number2, MemorySpaceType>::
        copy(thread_loop_partitioner,
             partitioner->locally_owned_size(),
             src.data,
             data);
    }
}



template <typename Number, typename MemorySpaceType>
template <typename MemorySpaceType2>
void
GhostedVector<Number, MemorySpaceType>::import(
  const GhostedVector<Number, MemorySpaceType2> &src,
  VectorOperation::values                        operation)
{
  Assert(src.partitioner.get() != nullptr, ExcNotInitialized());
  Assert(partitioner->locally_owned_range() ==
           src.partitioner->locally_owned_range(),
         ExcMessage("Locally owned indices should be identical."));
  Assert(partitioner->ghost_indices() == src.partitioner->ghost_indices(),
         ExcMessage("Ghost indices should be identical."));
  internal::VectorOperations::functions<Number, Number, MemorySpaceType>::
    import_elements(
      thread_loop_partitioner, allocated_size, operation, src.data, data);
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::compress(
  VectorOperation::values operation)
{
  compress_start(0, operation);
  compress_finish(operation);
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::update_ghost_values() const
{
  update_ghost_values_start();
  update_ghost_values_finish();
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::zero_out_ghosts() const
{
  this->zero_out_ghost_values();
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::zero_out_ghost_values() const
{
  if (data.values != nullptr)
    std::fill_n(data.values.get() + partitioner->locally_owned_size(),
                partitioner->n_ghost_indices(),
                Number());
#ifdef DEAL_II_COMPILER_CUDA_AWARE
  if (data.values_dev != nullptr)
    {
      const cudaError_t cuda_error_code =
        cudaMemset(data.values_dev.get() + partitioner->locally_owned_size(),
                   0,
                   partitioner->n_ghost_indices() * sizeof(Number));
      AssertCuda(cuda_error_code);
    }
#endif

  vector_is_ghosted = false;
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::compress_start(
  const unsigned int      communication_channel,
  VectorOperation::values operation)
{
  AssertIndexRange(communication_channel, 200);
  Assert(vector_is_ghosted == false,
         ExcMessage("Cannot call compress() on a ghosted vector"));

#ifdef DEAL_II_WITH_MPI
  // make this function thread safe
  std::lock_guard<std::mutex> lock(mutex);

  // allocate import_data in case it is not set up yet
  if (partitioner->n_import_indices() > 0)
    {
#  if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
    defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
      if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
        {
          if (import_data.values_dev == nullptr)
            import_data.values_dev.reset(
              Utilities::CUDA::allocate_device_data<Number>(
                partitioner->n_import_indices()));
        }
      else
#  endif
        {
#  if !defined(DEAL_II_COMPILER_CUDA_AWARE) && \
    defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
          static_assert(
            std::is_same<MemorySpaceType, MemorySpace::Host>::value,
            "This code path should only be compiled for CUDA-aware-MPI for MemorySpace::Host!");
#  endif
          if (import_data.values == nullptr)
            {
              Number *new_val;
              Utilities::System::posix_memalign(
                reinterpret_cast<void **>(&new_val),
                64,
                sizeof(Number) * partitioner->n_import_indices());
              import_data.values.reset(new_val);
            }
        }
    }

#  if defined DEAL_II_COMPILER_CUDA_AWARE && \
    !defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
  if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
    {
      // Move the data to the host and then move it back to the
      // device. We use values to store the elements because the function
      // uses a view of the array and thus we need the data on the host to
      // outlive the scope of the function.
      Number *new_val;
      Utilities::System::posix_memalign(reinterpret_cast<void **>(&new_val),
                                        64,
                                        sizeof(Number) * allocated_size);

      data.values = {new_val, [](Number *data) { std::free(data); }};

      cudaError_t cuda_error_code = cudaMemcpy(data.values.get(),
                                               data.values_dev.get(),
                                               allocated_size * sizeof(Number),
                                               cudaMemcpyDeviceToHost);
      AssertCuda(cuda_error_code);
    }
#  endif

#  if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
    defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
  if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
    {
      partitioner->import_from_ghosted_array_start(
        operation,
        communication_channel,
        ArrayView<Number, MemorySpace::CUDA>(
          data.values_dev.get() + partitioner->locally_owned_size(),
          partitioner->n_ghost_indices()),
        ArrayView<Number, MemorySpace::CUDA>(import_data.values_dev.get(),
                                             partitioner->n_import_indices()),
        compress_requests);
    }
  else
#  endif
    {
      partitioner->import_from_ghosted_array_start(
        operation,
        communication_channel,
        ArrayView<Number, MemorySpace::Host>(
          data.values.get() + partitioner->locally_owned_size(),
          partitioner->n_ghost_indices()),
        ArrayView<Number, MemorySpace::Host>(import_data.values.get(),
                                             partitioner->n_import_indices()),
        compress_requests);
    }
#else
  (void)communication_channel;
  (void)operation;
#endif
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::compress_finish(
  VectorOperation::values operation)
{
#ifdef DEAL_II_WITH_MPI
  vector_is_ghosted = false;

  // in order to zero ghost part of the vector, we need to call
  // import_from_ghosted_array_finish() regardless of
  // compress_requests.size() == 0

  // make this function thread safe
  std::lock_guard<std::mutex> lock(mutex);
#  if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
    defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
  if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
    {
      Assert(partitioner->n_import_indices() == 0 ||
               import_data.values_dev != nullptr,
             ExcNotInitialized());
      partitioner->import_from_ghosted_array_finish<Number, MemorySpace::CUDA>(
        operation,
        ArrayView<const Number, MemorySpace::CUDA>(
          import_data.values_dev.get(), partitioner->n_import_indices()),
        ArrayView<Number, MemorySpace::CUDA>(data.values_dev.get(),
                                             partitioner->locally_owned_size()),
        ArrayView<Number, MemorySpace::CUDA>(
          data.values_dev.get() + partitioner->locally_owned_size(),
          partitioner->n_ghost_indices()),
        compress_requests);
    }
  else
#  endif
    {
      Assert(partitioner->n_import_indices() == 0 ||
               import_data.values != nullptr,
             ExcNotInitialized());
      partitioner->import_from_ghosted_array_finish<Number, MemorySpace::Host>(
        operation,
        ArrayView<const Number, MemorySpace::Host>(
          import_data.values.get(), partitioner->n_import_indices()),
        ArrayView<Number, MemorySpace::Host>(data.values.get(),
                                             partitioner->locally_owned_size()),
        ArrayView<Number, MemorySpace::Host>(
          data.values.get() + partitioner->locally_owned_size(),
          partitioner->n_ghost_indices()),
        compress_requests);
    }

#  if defined DEAL_II_COMPILER_CUDA_AWARE && \
    !defined  DEAL_II_MPI_WITH_CUDA_SUPPORT
  // The communication is done on the host, so we need to
  // move the data back to the device.
  if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
    {
      cudaError_t cuda_error_code = cudaMemcpy(data.values_dev.get(),
                                               data.values.get(),
                                               allocated_size * sizeof(Number),
                                               cudaMemcpyHostToDevice);
      AssertCuda(cuda_error_code);

      data.values.reset();
    }
#  endif
#else
  (void)operation;
#endif
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::update_ghost_values_start(
  const unsigned int communication_channel) const
{
  AssertIndexRange(communication_channel, 200);
#ifdef DEAL_II_WITH_MPI
  // nothing to do when we neither have import nor ghost indices.
  if (partitioner->n_ghost_indices() == 0 &&
      partitioner->n_import_indices() == 0)
    return;

  // make this function thread safe
  std::lock_guard<std::mutex> lock(mutex);

  // allocate import_data in case it is not set up yet
  if (partitioner->n_import_indices() > 0)
    {
#  if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
    defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
      Assert(
        (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value),
        ExcMessage(
          "Using MemorySpace::CUDA only allowed if the code is compiled with a CUDA compiler!"));
      if (import_data.values_dev == nullptr)
        import_data.values_dev.reset(
          Utilities::CUDA::allocate_device_data<Number>(
            partitioner->n_import_indices()));
#  else
#    ifdef DEAL_II_MPI_WITH_CUDA_SUPPORT
      static_assert(
        std::is_same<MemorySpaceType, MemorySpace::Host>::value,
        "This code path should only be compiled for CUDA-aware-MPI for MemorySpace::Host!");
#    endif
      if (import_data.values == nullptr)
        {
          Number *new_val;
          Utilities::System::posix_memalign(reinterpret_cast<void **>(&new_val),
                                            64,
                                            sizeof(Number) *
                                              partitioner->n_import_indices());
          import_data.values.reset(new_val);
        }
#  endif
    }

#  if defined DEAL_II_COMPILER_CUDA_AWARE && \
    !defined(DEAL_II_MPI_WITH_CUDA_SUPPORT)
  // Move the data to the host and then move it back to the
  // device. We use values to store the elements because the function
  // uses a view of the array and thus we need the data on the host to
  // outlive the scope of the function.
  Number *new_val;
  Utilities::System::posix_memalign(reinterpret_cast<void **>(&new_val),
                                    64,
                                    sizeof(Number) * allocated_size);

  data.values = {new_val, [](Number *data) { std::free(data); }};

  cudaError_t cuda_error_code = cudaMemcpy(data.values.get(),
                                           data.values_dev.get(),
                                           allocated_size * sizeof(Number),
                                           cudaMemcpyDeviceToHost);
  AssertCuda(cuda_error_code);
#  endif

#  if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
        defined(DEAL_II_MPI_WITH_CUDA_SUPPORT))
  partitioner->export_to_ghosted_array_start<Number, MemorySpace::Host>(
    communication_channel,
    ArrayView<const Number, MemorySpace::Host>(
      data.values.get(), partitioner->locally_owned_size()),
    ArrayView<Number, MemorySpace::Host>(import_data.values.get(),
                                         partitioner->n_import_indices()),
    ArrayView<Number, MemorySpace::Host>(data.values.get() +
                                           partitioner->locally_owned_size(),
                                         partitioner->n_ghost_indices()),
    update_ghost_values_requests);
#  else
  partitioner->export_to_ghosted_array_start<Number, MemorySpace::CUDA>(
    communication_channel,
    ArrayView<const Number, MemorySpace::CUDA>(
      data.values_dev.get(), partitioner->locally_owned_size()),
    ArrayView<Number, MemorySpace::CUDA>(import_data.values_dev.get(),
                                         partitioner->n_import_indices()),
    ArrayView<Number, MemorySpace::CUDA>(data.values_dev.get() +
                                           partitioner->locally_owned_size(),
                                         partitioner->n_ghost_indices()),
    update_ghost_values_requests);
#  endif

#else
  (void)communication_channel;
#endif
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::update_ghost_values_finish() const
{
#ifdef DEAL_II_WITH_MPI
  // wait for both sends and receives to complete, even though only
  // receives are really necessary. this gives (much) better performance
  AssertDimension(partitioner->ghost_targets().size() +
                    partitioner->import_targets().size(),
                  update_ghost_values_requests.size());
  if (update_ghost_values_requests.size() > 0)
    {
      // make this function thread safe
      std::lock_guard<std::mutex> lock(mutex);

#  if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
        defined(DEAL_II_MPI_WITH_CUDA_SUPPORT))
      partitioner->export_to_ghosted_array_finish(
        ArrayView<Number, MemorySpace::Host>(
          data.values.get() + partitioner->locally_owned_size(),
          partitioner->n_ghost_indices()),
        update_ghost_values_requests);
#  else
      partitioner->export_to_ghosted_array_finish(
        ArrayView<Number, MemorySpace::CUDA>(
          data.values_dev.get() + partitioner->locally_owned_size(),
          partitioner->n_ghost_indices()),
        update_ghost_values_requests);
#  endif
    }

#  if defined DEAL_II_COMPILER_CUDA_AWARE && \
    !defined  DEAL_II_MPI_WITH_CUDA_SUPPORT
  // The communication is done on the host, so we need to
  // move the data back to the device.
  if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
    {
      cudaError_t cuda_error_code =
        cudaMemcpy(data.values_dev.get() + partitioner->locally_owned_size(),
                   data.values.get() + partitioner->locally_owned_size(),
                   partitioner->n_ghost_indices() * sizeof(Number),
                   cudaMemcpyHostToDevice);
      AssertCuda(cuda_error_code);

      data.values.reset();
    }
#  endif

#endif
  vector_is_ghosted = true;
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::swap(
  GhostedVector<Number, MemorySpaceType> &v)
{
#ifdef DEAL_II_WITH_MPI

#  ifdef DEBUG
  if (Utilities::MPI::job_supports_mpi())
    {
      // make sure that there are not outstanding requests from updating
      // ghost values or compress
      int flag = 1;
      if (update_ghost_values_requests.size() > 0)
        {
          const int ierr = MPI_Testall(update_ghost_values_requests.size(),
                                       update_ghost_values_requests.data(),
                                       &flag,
                                       MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
          Assert(flag == 1,
                 ExcMessage(
                   "MPI found unfinished update_ghost_values() requests "
                   "when calling swap, which is not allowed."));
        }
      if (compress_requests.size() > 0)
        {
          const int ierr = MPI_Testall(compress_requests.size(),
                                       compress_requests.data(),
                                       &flag,
                                       MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
          Assert(flag == 1,
                 ExcMessage("MPI found unfinished compress() requests "
                            "when calling swap, which is not allowed."));
        }
    }
#  endif

  std::swap(compress_requests, v.compress_requests);
  std::swap(update_ghost_values_requests, v.update_ghost_values_requests);
#endif

  std::swap(partitioner, v.partitioner);
  std::swap(thread_loop_partitioner, v.thread_loop_partitioner);
  std::swap(allocated_size, v.allocated_size);
  std::swap(data, v.data);
  std::swap(import_data, v.import_data);
  std::swap(vector_is_ghosted, v.vector_is_ghosted);
}



template <typename Number, typename MemorySpaceType>
inline bool
GhostedVector<Number, MemorySpaceType>::partitioners_are_compatible(
  const Utilities::MPI::Partitioner &part) const
{
  return partitioner->is_compatible(part);
}



template <typename Number, typename MemorySpaceType>
inline bool
GhostedVector<Number, MemorySpaceType>::partitioners_are_globally_compatible(
  const Utilities::MPI::Partitioner &part) const
{
  return partitioner->is_globally_compatible(part);
}



template <typename Number, typename MemorySpaceType>
std::size_t
GhostedVector<Number, MemorySpaceType>::memory_consumption() const
{
  std::size_t memory = sizeof(*this);
  memory += sizeof(Number) * static_cast<std::size_t>(allocated_size);

  // if the partitioner is shared between more processors, just count a
  // fraction of that memory, since we're not actually using more memory
  // for it.
  if (partitioner.use_count() > 0)
    memory += partitioner->memory_consumption() / partitioner.use_count() + 1;
  if (import_data.values != nullptr || import_data.values_dev != nullptr)
    memory += (static_cast<std::size_t>(partitioner->n_import_indices()) *
               sizeof(Number));
  return memory;
}



template <typename Number, typename MemorySpaceType>
void
GhostedVector<Number, MemorySpaceType>::print(std::ostream &     out,
                                              const unsigned int precision,
                                              const bool         scientific,
                                              const bool         across) const
{
  Assert(partitioner.get() != nullptr, ExcInternalError());
  AssertThrow(out, ExcIO());
  std::ios::fmtflags old_flags     = out.flags();
  unsigned int       old_precision = out.precision(precision);

  out.precision(precision);
  if (scientific)
    out.setf(std::ios::scientific, std::ios::floatfield);
  else
    out.setf(std::ios::fixed, std::ios::floatfield);

    // to make the vector write out all the information in order, use as
    // many barriers as there are processors and start writing when it's our
    // turn
#ifdef DEAL_II_WITH_MPI
  if (partitioner->n_mpi_processes() > 1)
    for (unsigned int i = 0; i < partitioner->this_mpi_process(); i++)
      {
        const int ierr = MPI_Barrier(partitioner->get_mpi_communicator());
        AssertThrowMPI(ierr);
      }
#endif

  std::vector<Number> stored_elements(allocated_size);
  data.copy_to(stored_elements.data(), allocated_size);

  out << "Process #" << partitioner->this_mpi_process() << std::endl
      << "Local range: [" << partitioner->local_range().first << ", "
      << partitioner->local_range().second
      << "), global size: " << partitioner->size() << std::endl
      << "Vector data:" << std::endl;
  if (across)
    for (size_type i = 0; i < partitioner->locally_owned_size(); ++i)
      out << stored_elements[i] << ' ';
  else
    for (size_type i = 0; i < partitioner->locally_owned_size(); ++i)
      out << stored_elements[i] << std::endl;
  out << std::endl;

  if (vector_is_ghosted)
    {
      out << "Ghost entries (global index / value):" << std::endl;
      if (across)
        for (size_type i = 0; i < partitioner->n_ghost_indices(); ++i)
          out << '(' << partitioner->ghost_indices().nth_index_in_set(i) << '/'
              << stored_elements[partitioner->locally_owned_size() + i] << ") ";
      else
        for (size_type i = 0; i < partitioner->n_ghost_indices(); ++i)
          out << '(' << partitioner->ghost_indices().nth_index_in_set(i) << '/'
              << stored_elements[partitioner->locally_owned_size() + i] << ")"
              << std::endl;
      out << std::endl;
    }
  out << std::flush;

#ifdef DEAL_II_WITH_MPI
  if (partitioner->n_mpi_processes() > 1)
    {
      int ierr = MPI_Barrier(partitioner->get_mpi_communicator());
      AssertThrowMPI(ierr);

      for (unsigned int i = partitioner->this_mpi_process() + 1;
           i < partitioner->n_mpi_processes();
           i++)
        {
          ierr = MPI_Barrier(partitioner->get_mpi_communicator());
          AssertThrowMPI(ierr);
        }
    }
#endif

  AssertThrow(out, ExcIO());
  // reset output format
  out.flags(old_flags);
  out.precision(old_precision);
}



DEAL_II_NAMESPACE_CLOSE

#endif
