using Palmprint.Application.DTOs;
using Palmprint.Application.Interfaces;

namespace Palmprint.Application.Services;

public class IdentificationService : IIdentificationService
{
	private const double ScoreThreshold = 0.93;
	private const double GapThreshold = 0.02;
	private const int TopK = 5;

	private readonly IIdentificationRepository _repository;
	private readonly IAiServiceClient _aiServiceClient;
	private readonly ITemplateSecurityService _templateSecurityService;
	private readonly IOperationLogger _operationLogger;

	public IdentificationService(
		IIdentificationRepository repository,
		IAiServiceClient aiServiceClient,
		ITemplateSecurityService templateSecurityService,
		IOperationLogger operationLogger)
	{
		_repository = repository;
		_aiServiceClient = aiServiceClient;
		_templateSecurityService = templateSecurityService;
		_operationLogger = operationLogger;
	}

	public async Task<IdentifyResponse> IdentifyAsync(IdentifyRequest request)
	{
		if (request.TenantId == Guid.Empty)
			throw new ArgumentException("TenantId is required.");

		if (request.ImageBytes.Length == 0)
			throw new ArgumentException("Image is required.");

		var aiResult = await _aiServiceClient.GenerateEmbeddingAsync(request.ImageBytes);

		var queryCnnEmbedding = aiResult.CnnEmbedding.ToArray();
		var queryTripletEmbedding = aiResult.TripletEmbedding.ToArray();

		NormalizeInPlace(queryCnnEmbedding);
		NormalizeInPlace(queryTripletEmbedding);

		var templates = await _repository.GetActiveTemplatesByTenantAsync(request.TenantId);

		if (templates.Count == 0)
		{
			await _operationLogger.LogAsync(
				request.TenantId,
				null,
				"IDENTIFICATION",
				false,
				null,
				"No active templates found for this tenant.",
				null
			);

			return new IdentifyResponse
			{
				Success = false,
				SimilarityScore = 0,
				ScoreGap = 0,
				Threshold = ScoreThreshold,
				GapThreshold = GapThreshold,
				Message = "No active templates found for this tenant."
			};
		}

		var candidates = new List<IdentifyCandidateDto>();

		foreach (var template in templates)
		{
			if (template.User == null)
				continue;

			var storedEmbedding = _templateSecurityService.DecryptEmbedding(template.EncryptedEmbedding);

			if (storedEmbedding.Length != 1792)
				continue;

			var storedCnnEmbedding = storedEmbedding.Take(1280).ToArray();
			var storedTripletEmbedding = storedEmbedding.Skip(1280).Take(512).ToArray();

			NormalizeInPlace(storedCnnEmbedding);
			NormalizeInPlace(storedTripletEmbedding);

			var cnnScore = CosineSimilarity(queryCnnEmbedding, storedCnnEmbedding);
			var tripletScore = CosineSimilarity(queryTripletEmbedding, storedTripletEmbedding);

			var score = 0.42 * cnnScore + 0.58 * tripletScore;

			candidates.Add(new IdentifyCandidateDto
			{
				UserId = template.User.Id,
				FullName = template.User.FullName,
				ExternalId = template.User.ExternalId,
				Score = score
			});
		}

		var identityCandidates = candidates
			.GroupBy(x => x.UserId)
			.Select(g =>
			{
				var bestCandidate = g.OrderByDescending(x => x.Score).First();

				return new IdentifyCandidateDto
				{
					UserId = bestCandidate.UserId,
					FullName = bestCandidate.FullName,
					ExternalId = bestCandidate.ExternalId,
					Score = bestCandidate.Score
				};
			})
			.OrderByDescending(x => x.Score)
			.ToList();

		var topCandidates = identityCandidates
			.Take(TopK)
			.ToList();

		var best = topCandidates.FirstOrDefault();

		if (best == null)
		{
			return new IdentifyResponse
			{
				Success = false,
				SimilarityScore = 0,
				ScoreGap = 0,
				Threshold = ScoreThreshold,
				GapThreshold = GapThreshold,
				Message = "No valid candidates found."
			};
		}

		var second = topCandidates.Skip(1).FirstOrDefault();
		var scoreGap = second == null ? best.Score : best.Score - second.Score;

		var accepted = best.Score >= ScoreThreshold && scoreGap >= GapThreshold;

		var message = accepted
			? "Identification completed successfully."
			: $"Identification rejected. Score={best.Score:F4}, Gap={scoreGap:F4}.";

		await _operationLogger.LogAsync(
			request.TenantId,
			accepted ? best.UserId : null,
			"IDENTIFICATION",
			accepted,
			best.Score,
			message,
			null
		);

		return new IdentifyResponse
		{
			Success = accepted,
			UserId = accepted ? best.UserId : null,
			FullName = accepted ? best.FullName : null,
			ExternalId = accepted ? best.ExternalId : null,
			SimilarityScore = best.Score,
			ScoreGap = scoreGap,
			Threshold = ScoreThreshold,
			GapThreshold = GapThreshold,
			Message = message,
			TopCandidates = topCandidates
		};
	}

	private static double CosineSimilarity(float[] a, float[] b)
	{
		if (a.Length != b.Length)
			throw new InvalidOperationException($"Embedding size mismatch: {a.Length} vs {b.Length}");

		double dot = 0.0;

		for (int i = 0; i < a.Length; i++)
			dot += a[i] * b[i];

		return dot;
	}

	private static void NormalizeInPlace(float[] vector)
	{
		double norm = 0.0;

		for (int i = 0; i < vector.Length; i++)
			norm += vector[i] * vector[i];

		norm = Math.Sqrt(norm);

		if (norm <= 1e-12)
			return;

		for (int i = 0; i < vector.Length; i++)
			vector[i] = (float)(vector[i] / norm);
	}
}