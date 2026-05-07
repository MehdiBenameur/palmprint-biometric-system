namespace Palmprint.Application.Interfaces;

public interface IAiServiceClient
{
    Task<(float[] Embedding, double QualityScore, string ModelVersion)> GenerateEmbeddingAsync(byte[] imageBytes);
}