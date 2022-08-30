import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.stream.Collectors;

import slp.core.counting.giga.GigaCounter;
import slp.core.lexing.Lexer;
import slp.core.lexing.code.JavaLexer;
import slp.core.lexing.runners.LexerRunner;
import slp.core.lexing.simple.WhitespaceLexer;
import slp.core.modeling.Model;
import slp.core.modeling.dynamic.CacheModel;
import slp.core.modeling.mix.MixModel;
import slp.core.modeling.ngram.JMModel;
import slp.core.modeling.runners.ModelRunner;
import slp.core.translating.Vocabulary;

public class n_gram {

    public static ModelRunner train_model() {
        Map to_return = new HashMap();
        File train = new File("../../data/ngram/train_data.txt");
        Lexer lexer = new WhitespaceLexer();   // Use a Java lexer; if your code is already lexed, use whitespace or tokenized lexer
        LexerRunner lexerRunner = new LexerRunner(lexer, false);

        lexerRunner.setSentenceMarkers(true);  // Add start and end markers to the files

        Vocabulary vocabulary = new Vocabulary();  // Create an empty vocabulary

        Model model = new JMModel(6, new GigaCounter());  // Standard smoothing for code, giga-counter for large corpora
        model = MixModel.standard(model, new CacheModel());  // Use a simple cache model; see JavaRunner for more options
        ModelRunner modelRunner = new ModelRunner(model, lexerRunner, vocabulary); // Use above lexer and vocabulary
//        modelRunner.learnDirectory(train);  // Teach the model all the data in "train"
        modelRunner.learnFile(train);

        return modelRunner;
    }

    public static void predict_defective_lines(ModelRunner modelRunner) throws Exception {
        LexerRunner lexerRunner = modelRunner.getLexerRunner();

        StringBuilder sb = new StringBuilder();

        sb.append("commit-id\tline-idx\tline-score\tlabel\n");
        File test = new File("../../data/ngram/test_data_line.txt");
        String commits_path = "../../data/ngram/test_data_commit.txt";
        String ids_path = "../../data/ngram/test_data_id.txt";
        String labels_path = "../../data/ngram/test_data_label.txt";

//        String project_path = root_dir + "/projects.txt";

        // loop each file here...

        List<String> commits = FileUtils.readLines(new File(commits_path), "UTF-8");
        List<String> ids = FileUtils.readLines(new File(ids_path), "UTF-8");
        List<String> labels = FileUtils.readLines(new File(labels_path), "UTF-8");
//        List<String> projects = FileUtils.readLines(new File(project_path), "UTF-8");

        System.out.println("commit: " + commits.size());
        List<List<Double>> fileEntropies = modelRunner.modelFile(test);
//        List<List<String>> fileTokens = lexerRunner.lexFile(test)  // Let's also retrieve the tokens on each line
//                .map(l -> l.collect(Collectors.toList()))
//                .collect(Collectors.toList());

        for (int i = 0; i < commits.size(); i++) {
//            List<String> lineTokens = fileTokens.get(i);
            List<Double> lineEntropies = fileEntropies.get(i);

            String commit_id = commits.get(i);
            String line_id = ids.get(i);
            String label = labels.get(i);

//            String project_name = projects.get(i);


            // First use Java's stream API to summarize entropies on this line
            // (see modelRunner.getStats for summarizing file or directory results)
            DoubleSummaryStatistics lineStatistics = lineEntropies.stream()
                    .mapToDouble(Double::doubleValue)
                    .summaryStatistics();
            double averageEntropy = lineStatistics.getAverage();

            sb.append(commit_id + "\t" + line_id + "\t" + averageEntropy + "\t" + label + "\n");


        }
       FileUtils.write(new File("./line-level-result-onlyadds.txt"), sb.toString(), "UTF-8");
    }

    public static void train_eval_model() throws Exception {

        ModelRunner modelRunner = train_model();

        System.out.println("finish training model ...");

        predict_defective_lines(modelRunner);

    }

    public static void main(String[] args) throws Exception {
        train_eval_model();
    }
}
