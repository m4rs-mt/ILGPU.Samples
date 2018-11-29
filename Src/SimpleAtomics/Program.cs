// -----------------------------------------------------------------------------
//                                ILGPU Samples
//                   Copyright (c) 2017 ILGPU Samples Project
//                                www.ilgpu.net
//
// File: Program.cs
//
// This file is part of ILGPU and is distributed under the University of
// Illinois Open Source License. See LICENSE.txt for details.
// -----------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;
using System;
using System.Reflection;

namespace SimpleAtomics
{
    class Program
    {
        /// <summary>
        /// A simple 1D kernel using basic atomic functions.
        /// The second parameter (<paramref name="dataView"/>) represents the target
        /// view for all atomic operations.
        /// </summary>
        /// <param name="index">The current thread index.</param>
        /// <param name="dataView">The view pointing to our memory buffer.</param>
        /// <param name="constant">A uniform constant.</param>
        static void AtomicOperationKernel(
            Index index,               // The global thread index (1D in this case)
            ArrayView<int> dataView,   // A view to a chunk of memory (1D in this case)
            int constant)              // A sample uniform constant
        {
            // dataView[0] += constant
            Atomic.Add(ref dataView.GetVariableView(0).Value, constant);

            // dataView[1] -= constant
            Atomic.Add(ref dataView.GetVariableView(1).Value, -constant);

            // dataView[2] = Max(dataView[2], constant)
            Atomic.Max(ref dataView.GetVariableView(2).Value, constant);

            // dataView[3] = Min(dataView[3], constant)
            Atomic.Min(ref dataView.GetVariableView(3).Value, constant);

            // dataView[4] = Min(dataView[4], constant)
            Atomic.And(ref dataView.GetVariableView(4).Value, constant);

            // dataView[5] = Min(dataView[5], constant)
            Atomic.Or(ref dataView.GetVariableView(5).Value, constant);

            // dataView[6] = Min(dataView[6], constant)
            Atomic.Xor(ref dataView.GetVariableView(6).Value, constant);
        }

        /// <summary>
        /// Launches a simple 1D kernel.
        /// </summary>
        static void Main(string[] args)
        {
            // Create main context
            using (var context = new Context())
            {
                // For each available accelerator...
                foreach (var acceleratorId in Accelerator.Accelerators)
                {
                    // Create default accelerator for the given accelerator id
                    using (var accelerator = Accelerator.Create(context, acceleratorId))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                        var kernel = accelerator.LoadAutoGroupedKernel<
                            Index, ArrayView<int>, int>(AtomicOperationKernel);
                        using (var buffer = accelerator.Allocate<int>(7))
                        {
                            // Initialize buffer to zero
                            buffer.MemSetToZero();

                            // Launch buffer.Length many threads and pass a view to buffer
                            kernel(accelerator.DefaultStream, 1024, buffer.View, 4);

                            // Wait for the kernel to finish...
                            accelerator.Synchronize();

                            // Resolve data
                            var data = buffer.GetAsArray(accelerator.DefaultStream);
                            for (int i = 0, e = data.Length; i < e; ++i)
                                Console.WriteLine($"Data[{i}] = {data[i]}");
                        }
                    }
                }
            }
        }
    }
}
