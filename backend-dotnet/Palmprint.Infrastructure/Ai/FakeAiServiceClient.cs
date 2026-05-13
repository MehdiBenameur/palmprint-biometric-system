using Palmprint.Application.Interfaces;

namespace Palmprint.Infrastructure.Ai;

public class FakeAiServiceClient : IAiServiceClient
{
    public Task<(float[] CnnEmbedding, float[] TripletEmbedding, double QualityScore, string ModelVersion)> GenerateEmbeddingAsync(byte[] imageBytes)
    {
        var random = new Random();

        var cnnEmbedding = new float[1280];
        var tripletEmbedding = new float[512];

        for (int i = 0; i < cnnEmbedding.Length; i++)
            cnnEmbedding[i] = (float)random.NextDouble();

        for (int i = 0; i < tripletEmbedding.Length; i++)
            tripletEmbedding[i] = (float)random.NextDouble();

        return Task.FromResult((
            cnnEmbedding,
            tripletEmbedding,
            0.90,
            "fake-cnn-triplet-v1"
        ));
    }
}